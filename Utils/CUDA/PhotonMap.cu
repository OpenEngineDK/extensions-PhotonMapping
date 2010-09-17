// Photon map class for CUDA
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Utils/CUDA/PhotonMap.h>
#include <Utils/CUDA/Utils.h>

#include <Logging/Logger.h>

#include <Utils/CUDA/Kernels/PhotonMapDeviceVars.h>
#include <Utils/CUDA/Kernels/PhotonSort.h>
#include <Utils/CUDA/Kernels/UpperNodeBoundingBox.h>
#include <Utils/CUDA/Kernels/UpperNodeSplit.hcu>
#include <Utils/CUDA/Kernels/UpperNodeChildren.hcu>

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {

            using namespace Kernels;

            PhotonMap::PhotonMap(unsigned int size) {
                MAX_BLOCKS = (activeCudaDevice.maxGridSize[0] + 1) / activeCudaDevice.maxThreadsDim[0];
                logger.info << "MAX_BLOCKS " << MAX_BLOCKS << logger.end;

                // Initialized timer
                cutCreateTimer(&timerID);

                // Allocate photons on GPU
                photons = PhotonNode(size);
                photons.CreateRandomData();
                cudaSafeMalloc(&photonOwners, size * sizeof(int));
                cudaSafeMalloc(&newOwners, size * sizeof(int));
                
                // Make room for the root node
                int approxSize = (2 * size / PhotonLowerNode::MAX_SIZE) - 1;
                upperNodes = PhotonUpperNode(approxSize);

                tempChildren = NodeChildren(size / PhotonLowerNode::MAX_SIZE);

                // Split vars
                scanConfig.op = CUDPP_ADD;
                scanConfig.datatype = CUDPP_UINT;
                scanConfig.algorithm = CUDPP_SCAN;
                scanConfig.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
                
                CUDPPResult res = cudppPlan(&scanHandle, scanConfig, size, 1, 0);
                if (CUDPP_SUCCESS != res)
                    throw Core::Exception("Error creating CUDPP scanPlan");

                sortConfig.op = CUDPP_ADD;
                sortConfig.datatype = CUDPP_FLOAT;
                sortConfig.algorithm = CUDPP_SORT_RADIX;
                sortConfig.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
                
                res = cudppPlan(&sortHandle, sortConfig, size, 1, 0);
                if (CUDPP_SUCCESS != res)
                    throw Core::Exception("Error creating CUDPP sortPlan");

                cudaSafeMalloc(&xIndices, size * sizeof(float));
                cudaSafeMalloc(&xSorted, size * sizeof(float4));
                cudaSafeMalloc(&xKeys, size * sizeof(float));
                cudaSafeMalloc(&yIndices, size * sizeof(float));
                cudaSafeMalloc(&ySorted, size * sizeof(float4));
                cudaSafeMalloc(&yKeys, size * sizeof(float));
                cudaSafeMalloc(&zIndices, size * sizeof(float));
                cudaSafeMalloc(&zSorted, size * sizeof(float4));
                cudaSafeMalloc(&zKeys, size * sizeof(float));

                cudaSafeMalloc(&leafSide, size * sizeof(int));
                cudaSafeMalloc(&leafPrefix, (size+1) * sizeof(int));
                cudaSafeMalloc(&splitSide, size * sizeof(int));
                cudaSafeMalloc(&prefixSum, (size+1) * sizeof(int));
                cudaSafeMalloc(&splitLeft, (size+1) * sizeof(int));
                cudaSafeMalloc(&splitAddrs, size * sizeof(int2));
                cudaSafeMalloc(&tempPhotonPos, size * sizeof(float4));
            }

            void PhotonMap::Create(){
                int childrenCreated = 2, leafsCreated = 0;
                int activeIndex = 0, activeRange = 1;
                int activePhotons = photons.size;
                int unhandledLeafs = 0;

                // Initialize kd root node data
                int2 i = make_int2(0, activePhotons);
                cudaMemcpy(upperNodes.photonInfo, &i, sizeof(int2), cudaMemcpyHostToDevice);
                upperNodes.size = 1;

                // Set all photons owner to node 0
                cudaMemset(photonOwners, 0, activePhotons * sizeof(int));

                // Sort photons into arrays
                SortPhotons();

                // Process upper nodes
                int level = 0, maxLevel = 7;
                while (childrenCreated != 0 && level < maxLevel){
                    ProcessUpperNodes(activeIndex, activeRange, unhandledLeafs, 
                                      leafsCreated, childrenCreated, activePhotons);
                    
                    logger.info << "Created " << childrenCreated << " children and " << leafsCreated << " leafs" << logger.end;

                    for (int i = -unhandledLeafs; i < activeRange; ++i)
                        logger.info << upperNodes.ToString(i + activeIndex) << logger.end;
                    

                    // Increment loop variables
                    activeIndex += activeRange + leafsCreated;
                    activeRange = childrenCreated;
                    unhandledLeafs = leafsCreated;
                    level++;
                }
                // Copy the rest of the photons to photon position
                cudaMemcpy(xSorted, photons.pos, 
                           activePhotons * sizeof(point), cudaMemcpyDeviceToDevice);
                for (int i = -unhandledLeafs; i < activeRange; ++i)
                    logger.info << upperNodes.ToString(i + activeIndex) << logger.end;
                logger.info << "photons.pos" << Utils::CUDA::Convert::ToString(photons.pos, photons.size) << logger.end;
                // Setup photon info for the last nodes.
                

                // Preprocess lower nodes.

                // Process lower nodes.


#ifdef OE_SAFE
                VerifyMap();
#endif
            }

            void PhotonMap::SortPhotons(){
                //logger.info << "Sort all photon" << logger.end;

                int size = photons.size;

                unsigned int blocks, threads;
                Calc1DKernelDimensions(size, blocks, threads);
                
                Indices<<<blocks, threads>>>(photons.pos, 
                                             xIndices, yIndices, zIndices, 
                                             xKeys, yKeys, zKeys, 
                                             size);
                
                cudppSort(sortHandle, xKeys, xIndices, sizeof(float), size);
                cudppSort(sortHandle, yKeys, yIndices, sizeof(float), size);
                cudppSort(sortHandle, zKeys, zIndices, sizeof(float), size);

                //START_TIMER(timerID);
                ScatterPhotons<<<blocks, threads>>>(photons.pos, 
                                                    xIndices, yIndices, zIndices,
                                                    xSorted, ySorted, zSorted,
                                                    size);

                //PRINT_TIMER(timerID,"Sort photons");
                CHECK_FOR_CUDA_ERROR();
            }

            void PhotonMap::ProcessUpperNodes(int activeIndex,
                                              int activeRange,
                                              int unhandledLeafs,
                                              int &leafsCreated,
                                              int &childrenCreated,
                                              int &activePhotons){

                logger.info << "=== Process " << activeRange << " Upper Nodes === with " << activePhotons << " photons" << logger.end;

                // Check that there is room for the new children
                if (upperNodes.maxSize < upperNodes.size + activeRange * 2)
                    upperNodes.Resize(upperNodes.size + activeRange * 2);

                // Copy bookkeeping to symbols
                cudaMemcpyToSymbol(photonNodes, &activePhotons, sizeof(int));
                cudaMemcpyToSymbol(activeNodeIndex, &activeIndex, sizeof(int));
                cudaMemcpyToSymbol(activeNodeRange, &activeRange, sizeof(int));
                CHECK_FOR_CUDA_ERROR();

                ComputeBoundingBox(activeIndex, activeRange);
                CHECK_FOR_CUDA_ERROR();

                SplitUpperNodePhotons(activeIndex, activeRange, unhandledLeafs, activePhotons);
                CHECK_FOR_CUDA_ERROR();

                CreateChildren(activeIndex, activeRange, activePhotons,
                               leafsCreated, childrenCreated);
                CHECK_FOR_CUDA_ERROR();
                
                // Update uppernode size
                upperNodes.size += leafsCreated + childrenCreated;
                
                CHECK_FOR_CUDA_ERROR();
            }
            
            /**
             * Use the sorted arrays to compute the bounding box in
             * constant time.
             */
            void PhotonMap::ComputeBoundingBox(int activeIndex,
                                               int activeRange){
                //logger.info << "Compute bounding boxes" << logger.end;
                
                unsigned int blocks, threads;
                Calc1DKernelDimensions(activeRange, blocks, threads);

                ConstantTimeBoundingBox<<<blocks, threads>>>(upperNodes.photonInfo + activeIndex,
                                                             upperNodes.aabbMin + activeIndex,
                                                             upperNodes.aabbMax + activeIndex,
                                                             xSorted, ySorted, zSorted);

            }
            
            void PhotonMap::SplitUpperNodePhotons(int activeIndex,
                                                  int activeRange,
                                                  int unhandledLeafs,
                                                  int &activePhotons){
                //logger.info << "Split Upper Node photons" << logger.end;
                
                unsigned int blocks, threads;
                Calc1DKernelDimensions(activeRange, blocks, threads);
                
                SetUpperNodeSplitInfo<<<blocks, threads>>>(upperNodes.aabbMin + activeIndex,
                                                           upperNodes.aabbMax + activeIndex,
                                                           upperNodes.splitPos + activeIndex,
                                                           upperNodes.info + activeIndex);

                //logger.info << "Root splitting plane " << upperNodes.ToString(0) << logger.end;
                
                Calc1DKernelDimensions(activePhotons, blocks, threads);

                // @OPT Only handle leafs nodes in case there
                // actually are any!
                // @OPT precompute leaf pos if there are leaf nodes?
                // Could remove to copy from symbols.

                // Set photons leaf bit
                SetPhotonNodeLeafSide<<<blocks, threads>>>(photons.pos, photonOwners, upperNodes.info, leafSide);
                CHECK_FOR_CUDA_ERROR();
                cudppScan(scanHandle, leafPrefix, leafSide, activePhotons);
                CHECK_FOR_CUDA_ERROR();

                /*
                logger.info << "activePhotons: " << activePhotons << logger.end;
                logger.info << "Leaf side: " << Utils::CUDA::Convert::ToString(leafSide, activePhotons) << logger.end;
                logger.info << "Leaf prefix/left: " << Utils::CUDA::Convert::ToString(leafPrefix, activePhotons) << logger.end;
                logger.info << "Owner: " << Utils::CUDA::Convert::ToString(photonOwners, activePhotons) << logger.end;
                */
                
                SetPhotonNodeSplitSide<<<blocks, threads>>>(xSorted, photonOwners, 
                                                            upperNodes.splitPos, upperNodes.info, 
                                                            splitSide);
                cudppScan(scanHandle, splitLeft, splitSide, activePhotons);
                CHECK_FOR_CUDA_ERROR();

                //logger.info << "xSorted: " << Utils::CUDA::Convert::ToString(xSorted, activePhotons) << logger.end;
                //logger.info << "Split side: " << Utils::CUDA::Convert::ToString(splitSide, activePhotons) << logger.end;
                
                SplitPhotons<<<blocks, threads>>>(xSorted, tempPhotonPos,
                                                  photonOwners, newOwners,
                                                  splitLeft, splitSide, 
                                                  leafPrefix, leafSide,
                                                  splitAddrs);
                CHECK_FOR_CUDA_ERROR();

                // Copy photon positions belonging to leaves to the photon nodes.
                int nonLeafPhotons;
                cudaMemcpy(&nonLeafPhotons, leafPrefix + activePhotons,
                           sizeof(int), cudaMemcpyDeviceToHost);
                int leafPhotons = activePhotons - nonLeafPhotons;
                if (leafPhotons > 0){
                    // Copy photons to a persistent array
                    cudaMemcpy(tempPhotonPos + nonLeafPhotons, photons.pos + nonLeafPhotons, 
                               leafPhotons * sizeof(point), cudaMemcpyDeviceToDevice);

                    // Setup the leafs nodes photon info
                }
                std::swap(xSorted, tempPhotonPos);

                SplitSortedArray(ySorted, activePhotons);
                SplitSortedArray(zSorted, activePhotons);

                std::swap(photonOwners, newOwners);

                if (unhandledLeafs > 0)
                    SetupUpperLeafNodes(activeIndex, unhandledLeafs, nonLeafPhotons);

                /*
                logger.info << "Updated Owners: " << Utils::CUDA::Convert::ToString(photonOwners, activePhotons) << logger.end;
                logger.info << "xSorted: " << Utils::CUDA::Convert::ToString(xSorted, photons.size) << logger.end;
                logger.info << "ySorted: " << Utils::CUDA::Convert::ToString(ySorted, photons.size) << logger.end;
                logger.info << "zSorted: " << Utils::CUDA::Convert::ToString(zSorted, photons.size) << logger.end;
                */
                activePhotons = nonLeafPhotons;

            }

            void PhotonMap::SplitSortedArray(float4 *&sortedArray, int activePhotons){
                unsigned int blocks, threads;
                Calc1DKernelDimensions(activePhotons, blocks, threads);

                // Calc splitting side for all sortedArray positions
                SetPhotonNodeSplitSide<<<blocks, threads>>>(sortedArray, photonOwners, upperNodes.splitPos, upperNodes.info, splitSide);
                cudppScan(scanHandle, prefixSum, splitSide, activePhotons);
                CHECK_FOR_CUDA_ERROR();

                SplitPhotons<<<blocks, threads>>>(sortedArray, tempPhotonPos, 
                                                  prefixSum, splitSide, 
                                                  leafPrefix, leafSide);
                CHECK_FOR_CUDA_ERROR();

                point* temp = sortedArray;
                sortedArray = tempPhotonPos;
                tempPhotonPos = temp;
            }
            
            void PhotonMap::SetupUpperLeafNodes(int activeIndex,
                                                int leafNodes,
                                                int photonOffset){

                //logger.info << "Setup " << leafNodes << " leaf nodes" << logger.end;
                
                // Update leafs nodes photon info.
                unsigned int blocks, threads;
                Calc1DKernelDimensions(leafNodes, blocks, threads);

                int leafIndex = activeIndex - leafNodes;
                //logger.info << "Leaf index " << leafIndex << logger.end;
                SetupLeafNodes<<<blocks, threads>>>(upperNodes.photonInfo + leafIndex,
                                                    leafPrefix,
                                                    leafNodes);
                CHECK_FOR_CUDA_ERROR();
                /*
                logger.info << "Leaf photon info" << 
                    Utils::CUDA::Convert::ToString(upperNodes.photonInfo + leafIndex, leafNodes) << logger.end;
                */
                // @TODO Create lower nodes

            }
            
            void PhotonMap::CreateChildren(int activeIndex,
                                           int activeRange,
                                           int activePhotons,
                                           int &leafsCreated,
                                           int &childrenCreated){
                //logger.info << "Create children to " << activeRange << " nodes" << logger.end;

                // @TODO extend with children counter (prefixSum) and
                // empty space splitting

                unsigned int blocks, threads;
                Calc1DKernelDimensions(activeRange, blocks, threads);

                // Allocate room for the temp children
                if (tempChildren.size < activeRange * 2){
                    tempChildren.Resize(activeRange * 2);
                }

                // Create children photon index and ranges in temp arrays.
                bool createdLeafs = false;
                cudaMemcpyToSymbol(d_createdLeafs, &createdLeafs, sizeof(bool));

                // Reuse leafSide to hold splits
                SetupChildren<<<blocks, threads>>>(upperNodes.photonInfo + activeIndex,
                                                   tempChildren.photonInfo, tempChildren.parents,
                                                   splitAddrs, splitLeft,
                                                   leafSide);
                CHECK_FOR_CUDA_ERROR();
                /*
                logger.info << "===" << logger.end;
                logger.info << "Photon info: " << Utils::CUDA::Convert::ToString(upperNodes.photonInfo + activeIndex, activeRange) << logger.end;
                logger.info << "SplitAddrs: " << Utils::CUDA::Convert::ToString(splitAddrs, activePhotons) << logger.end;
                logger.info << "Split left: " << Utils::CUDA::Convert::ToString(splitLeft, activePhotons+1) << logger.end;
                logger.info << "===" << logger.end;
                logger.info << "Temp Children photon info: " << Utils::CUDA::Convert::ToString(tempChildren.photonInfo, activeRange * 2) << logger.end;
                logger.info << "===" << logger.end;
                */
                cudaMemcpyFromSymbol(&createdLeafs, d_createdLeafs, sizeof(bool));

                if (createdLeafs){
                    // Sort leafs to the left
                    cudppScan(scanHandle, leafPrefix, leafSide, activeRange * 2);

                    //logger.info << Utils::CUDA::Convert::ToString(leafPrefix, activeRange * 2) << logger.end;
                    
                    // Update upper node children with index and range,
                    // and update parent with real address.
                    //logger.info << "Temp photon info: " << Utils::CUDA::Convert::ToString(tempChildren.photonInfo, activeRange * 2) << logger.end;
                    //logger.info << "split side: " << Utils::CUDA::Convert::ToString(leafSide, activeRange * 2) << logger.end;
                    //logger.info << "prefix sum: " << Utils::CUDA::Convert::ToString(leafPrefix, activeRange * 2) << logger.end;

                    MoveChildInfo<<<blocks, threads>>>(tempChildren.photonInfo,
                                                       leafSide, leafPrefix,
                                                       upperNodes.left + activeIndex, 
                                                       upperNodes.right + activeIndex,
                                                       upperNodes.photonInfo + activeIndex + activeRange,
                                                       upperNodes.info + activeIndex + activeRange);
                    //logger.info << "Child photon info: " << Utils::CUDA::Convert::ToString(upperNodes.photonInfo + activeIndex + activeRange, activeRange * 2) << logger.end;

                    CHECK_FOR_CUDA_ERROR();                    

                    cudaMemcpyFromSymbol(&leafsCreated, d_leafsCreated, sizeof(int));
                    CHECK_FOR_CUDA_ERROR();
                    childrenCreated = 2 * activeRange - leafsCreated;
                    
                }else{
                    // Move child info into uppernodes
                    CopyChildInfo<<<blocks, threads>>>(tempChildren.photonInfo,
                                                       upperNodes.left + activeIndex, 
                                                       upperNodes.right + activeIndex,
                                                       upperNodes.photonInfo + activeIndex + activeRange,
                                                       upperNodes.info + activeIndex + activeRange);
                    CHECK_FOR_CUDA_ERROR();

                    leafsCreated = 0;
                    childrenCreated = 2 * activeRange;
                }

                Calc1DKernelDimensions(activePhotons, blocks, threads);
                UpdatePhotonOwners<<<blocks, threads>>>(photonOwners, upperNodes.left,
                                                        upperNodes.right, upperNodes.photonInfo);
                CHECK_FOR_CUDA_ERROR();
            }
            
            void PhotonMap::PreprocessLowerNodes(int range){

            }
            
            void PhotonMap::ProcessLowerNodes(int activeIndex,
                                              int activeRange,
                                              int &childrenCreated){

            }
            
        }
    }
}
