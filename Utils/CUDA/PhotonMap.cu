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
#include <Utils/CUDA/Kernels/PreprocessLowerNodes.h>

#define CPU_VERIFY

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {

            using namespace Kernels;

            PhotonMap::PhotonMap(unsigned int size) {
                MAX_BLOCKS = activeCudaDevice.maxGridSize[0];
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
                lowerNodes = PhotonLowerNode(approxSize);

                tempChildren = NodeChildren(size / PhotonLowerNode::MAX_SIZE);
                upperNodeLeafList = UpperNodeLeafList(size / PhotonLowerNode::MAX_SIZE);

                // Split vars
                scanConfig.algorithm = CUDPP_SCAN;
                scanConfig.op = CUDPP_ADD;
                scanConfig.datatype = CUDPP_INT;
                scanConfig.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
                
                CUDPPResult res = cudppPlan(&scanHandle, scanConfig, size, 1, 0);
                if (CUDPP_SUCCESS != res)
                    throw Core::Exception("Error creating CUDPP scanPlan");

                sortConfig.algorithm = CUDPP_SORT_RADIX;
                sortConfig.datatype = CUDPP_FLOAT;
                sortConfig.options = CUDPP_OPTION_KEY_VALUE_PAIRS;
                
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

                cudaSafeMalloc(&leafSide, (size+1) * sizeof(int));
                cudaSafeMalloc(&leafPrefix, (size+1) * sizeof(int));
                cudaSafeMalloc(&splitSide, (size+1) * sizeof(int));
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
                int level = 0, maxLevel = -1;
                //START_TIMER(timerID);
                while (childrenCreated != 0 && level != maxLevel){
                    logger.info << "<<== PASS " << level << " ==>>" << logger.end;
                    logger.info << "Active index " << activeIndex << " and range " << activeRange << logger.end;
                    logger.info << "Active photons " << activePhotons << logger.end;

                    ProcessUpperNodes(activeIndex, activeRange, unhandledLeafs, 
                                      leafsCreated, childrenCreated, activePhotons);
                    
                    for (int i = -unhandledLeafs; i < activeRange; ++i)
                        logger.info << upperNodes.ToString(i + activeIndex) << logger.end;
                    

                    // Increment loop variables
                    activeIndex += activeRange + leafsCreated;
                    activeRange = childrenCreated;
                    unhandledLeafs = leafsCreated;
                    level++;

                    logger.info << "Created " << childrenCreated << " children and " << leafsCreated << " leafs" << logger.end;
                    logger.info << "Unhandled leafs " << unhandledLeafs << logger.end;
                }
                // Copy the rest of the photons to photon position
                cudaMemcpy(photons.pos, xSorted, 
                           activePhotons * sizeof(point), cudaMemcpyDeviceToDevice);
                
                unsigned int blocks, threads;
                Calc1DKernelDimensions(leafsCreated, blocks, threads);
                int leafIndex = activeIndex - unhandledLeafs;
                if (upperNodeLeafList.size + unhandledLeafs > upperNodeLeafList.maxSize)
                    upperNodeLeafList.Resize(upperNodeLeafList.size + unhandledLeafs);

                SetLeafNodeArrays<<<blocks, threads>>>(upperNodeLeafList.leafIDs + upperNodeLeafList.size,
                                                       leafIndex, unhandledLeafs);
                upperNodeLeafList.size += unhandledLeafs;

                for (int i = -unhandledLeafs; i < activeRange; ++i)
                    logger.info << upperNodes.ToString(i + activeIndex) << logger.end;
                //PRINT_TIMER(timerID, "Upper node creation");

                // logger.info << "photons.pos " << Utils::CUDA::Convert::ToString(photons.pos, photons.size) << logger.end;

                // @TODO Setup photon info for the last nodes. (Not
                // needed? Hasn't crashed yet)                

                // Preprocess lower nodes.
                PreprocessLowerNodes();
                
                // Process lower nodes.


#ifdef CPU_VERIFY
                VerifyMap();
#endif
            }

            void PhotonMap::SortPhotons(){
                //logger.info << "Sort all photon" << logger.end;

                //logger.info << "Photon pos: " << Utils::CUDA::Convert::ToString(photons.pos, photons.size) << logger.end;

                int size = photons.size;

                unsigned int blocks, threads;
                Calc1DKernelDimensions(size, blocks, threads);

                Indices<<<blocks, threads>>>(photons.pos, 
                                             xIndices, yIndices, zIndices, 
                                             xKeys, yKeys, zKeys, 
                                             size);
                CHECK_FOR_CUDA_ERROR();
                
                cudppSort(sortHandle, xKeys, xIndices, sizeof(float), size);
                cudppSort(sortHandle, yKeys, yIndices, sizeof(float), size);
                cudppSort(sortHandle, zKeys, zIndices, sizeof(float), size);

                //logger.info << "xIndices: " << Utils::CUDA::Convert::ToString(xIndices, 16) << logger.end;

                //START_TIMER(timerID);
                ScatterPhotons<<<blocks, threads>>>(photons.pos, 
                                                    xIndices, yIndices, zIndices,
                                                    xSorted, ySorted, zSorted,
                                                    size);

                //logger.info << "\nxSorted: " << Utils::CUDA::Convert::ToString(xSorted, 16) << logger.end;
                //logger.info << "\nySorted: " << Utils::CUDA::Convert::ToString(ySorted, 16) << logger.end;
                //logger.info << "\nzSorted: " << Utils::CUDA::Convert::ToString(zSorted, 16) << logger.end;

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
                cudaMemcpyToSymbol(d_photonNodes, &activePhotons, sizeof(int));
                cudaMemcpyToSymbol(d_activeNodeIndex, &activeIndex, sizeof(int));
                cudaMemcpyToSymbol(d_activeNodeRange, &activeRange, sizeof(int));
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

                SetUpperNodeSplitInfo<<<blocks, threads>>>(upperNodes.aabbMin + activeIndex,
                                                           upperNodes.aabbMax + activeIndex,
                                                           upperNodes.splitPos + activeIndex,
                                                           upperNodes.info + activeIndex);

                //logger.info << "aabbMin: " << Utils::CUDA::Convert::ToString(upperNodes.aabbMin + activeIndex, activeRange) << logger.end;
                //logger.info << "aabbMax: " << Utils::CUDA::Convert::ToString(upperNodes.aabbMax + activeIndex, activeRange) << logger.end;
#ifdef CPU_VERIFY
                // Check that the bounding box holds for all arrays
                for (int i = activeIndex; i < activeIndex + activeRange; ++i){
                    int2 photonInfo;
                    cudaMemcpy(&photonInfo, upperNodes.photonInfo + i, sizeof(int2), cudaMemcpyDeviceToHost);

                    point aabbMin, aabbMax;
                    cudaMemcpy(&aabbMin, upperNodes.aabbMin+i, sizeof(point), cudaMemcpyDeviceToHost);
                    cudaMemcpy(&aabbMax, upperNodes.aabbMax+i, sizeof(point), cudaMemcpyDeviceToHost);

                    point pos[photonInfo.y];
                    point cpuMin, cpuMax;
                    cudaMemcpy(pos, xSorted + photonInfo.x, photonInfo.y * sizeof(point), cudaMemcpyDeviceToHost);
                    cpuMin = cpuMax = pos[0];
                    for (int j = 0; j < photonInfo.y; ++j){
                        cpuMin = pointMin(cpuMin, pos[j]);
                        cpuMax = pointMax(cpuMax, pos[j]);
                    }
                    if (cpuMin.x != aabbMin.x && cpuMin.y != aabbMin.y && cpuMin.z != aabbMin.z)
                        throw Core::Exception("aabbMin error: CPU min " + Utils::CUDA::Convert::ToString(cpuMin)
                                              + ", GPU min " + Utils::CUDA::Convert::ToString(aabbMin));
                    if (cpuMax.x != aabbMax.x && cpuMax.y != aabbMax.y && cpuMax.z != aabbMax.z)
                        throw Core::Exception("aabbMax error");
                }
#endif
            }
            
            void PhotonMap::SplitUpperNodePhotons(int activeIndex,
                                                  int activeRange,
                                                  int unhandledLeafs,
                                                  int &activePhotons){
                //logger.info << "Split Upper Node photons" << logger.end;
                
                unsigned int blocks, threads;                
                Calc1DKernelDimensions(activePhotons, blocks, threads);

                // @OPT Only handle leafs nodes in case there actually
                // are any!  

                // @OPT Throw an extra 0 at the end of leafside, so we
                // can calculate the final left value in the prefix
                // sum aswell (and not have to do it in the next
                // kernel)

                // Set photons leaf bit
                SetPhotonNodeLeafSide<<<blocks, threads>>>(photonOwners, upperNodes.info, leafSide);
                CHECK_FOR_CUDA_ERROR();
                cudppScan(scanHandle, leafPrefix, leafSide, activePhotons+1);
                CHECK_FOR_CUDA_ERROR();

                SetPhotonNodeSplitSide<<<blocks, threads>>>(xSorted, photonOwners, 
                                                            upperNodes.splitPos, upperNodes.info, 
                                                            splitSide);
                cudppScan(scanHandle, splitLeft, splitSide, activePhotons+1);
                CHECK_FOR_CUDA_ERROR();
                
                cudaMemcpyToSymbol(d_nonLeafPhotons, leafPrefix + activePhotons, sizeof(int), 0, cudaMemcpyDeviceToDevice);
                cudaMemcpyToSymbol(d_photonsMovedLeft, splitLeft + activePhotons, sizeof(int), 0, cudaMemcpyDeviceToDevice);
                
                SplitPhotons<<<blocks, threads>>>(xSorted, tempPhotonPos,
                                                  photonOwners, newOwners,
                                                  splitLeft, splitSide, 
                                                  leafPrefix, leafSide,
                                                  splitAddrs);
                CHECK_FOR_CUDA_ERROR();
                /*
                logger.info << "LeafSide: " << Utils::CUDA::Convert::ToString(leafSide, activePhotons) << logger.end;
                logger.info << "LeafPrefix: " << Utils::CUDA::Convert::ToString(leafPrefix, activePhotons+1) << logger.end;

                logger.info << "split side: " << Utils::CUDA::Convert::ToString(splitSide, activePhotons) << logger.end;
                logger.info << "split left: " << Utils::CUDA::Convert::ToString(splitLeft, activePhotons+1) << logger.end;

                logger.info << "SplitAddrs: " << Utils::CUDA::Convert::ToString(splitAddrs, activePhotons) << logger.end;
                logger.info << "Addrs: " << Utils::CUDA::Convert::ToString(debug, activePhotons) << logger.end;

                logger.info << "Owners: " << Utils::CUDA::Convert::ToString(photonOwners, activePhotons) << logger.end;
                logger.info << "new owners: " << Utils::CUDA::Convert::ToString(newOwners, activePhotons) << logger.end;
                */                
                // Copy photon positions belonging to leaves to the photon nodes.
                int nonLeafPhotons;
                cudaMemcpy(&nonLeafPhotons, leafPrefix + activePhotons,
                           sizeof(int), cudaMemcpyDeviceToHost);
                int leafPhotons = activePhotons - nonLeafPhotons;
                if (leafPhotons > 0)
                    // Copy photons to a persistent array @OPT do it assync
                    cudaMemcpy(photons.pos + nonLeafPhotons, tempPhotonPos + nonLeafPhotons, 
                               leafPhotons * sizeof(point), cudaMemcpyDeviceToDevice);

                std::swap(xSorted, tempPhotonPos);

                SplitSortedArray(ySorted, activePhotons);
                SplitSortedArray(zSorted, activePhotons);

                /*
                logger.info << "Split xSorted: " << Utils::CUDA::Convert::ToString(xSorted, activePhotons) << logger.end;
                logger.info << "Split ySorted: " << Utils::CUDA::Convert::ToString(ySorted, activePhotons) << logger.end;
                logger.info << "Split zSorted: " << Utils::CUDA::Convert::ToString(zSorted, activePhotons) << logger.end;
                */
                std::swap(photonOwners, newOwners);

#ifdef CPU_VERIFY
                
#endif
                
                if (unhandledLeafs > 0)
                    SetupUpperLeafNodes(activeIndex, unhandledLeafs, nonLeafPhotons);

                activePhotons = nonLeafPhotons;

                //logger.info << "New active photons: " << activePhotons << logger.end;
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

                // Update leafs nodes photon info.
                unsigned int blocks, threads;
                Calc1DKernelDimensions(leafNodes, blocks, threads);

                int leafIndex = activeIndex - leafNodes;

                if (upperNodeLeafList.size + leafNodes < upperNodeLeafList.maxSize)
                    upperNodeLeafList.Resize(upperNodeLeafList.size + leafNodes);

                SetupLeafNodes<<<blocks, threads>>>(upperNodes.photonInfo + leafIndex,
                                                    leafPrefix,
                                                    upperNodeLeafList.leafIDs + upperNodeLeafList.size,
                                                    leafNodes);
                CHECK_FOR_CUDA_ERROR();

                upperNodeLeafList.size += leafNodes;

                // @TODO place node ids in an array in preperation for creating lower nodes.
            }
            
            void PhotonMap::CreateChildren(int activeIndex,
                                           int activeRange,
                                           int activePhotons,
                                           int &leafsCreated,
                                           int &childrenCreated){
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
                logger.info << "=== SetupChildren<<<" << blocks << ", " << threads << ">>> ===" << logger.end;
                logger.info << "Photon info: " << Utils::CUDA::Convert::ToString(upperNodes.photonInfo + activeIndex, activeRange) << logger.end;
                logger.info << "SplitAddrs: " << Utils::CUDA::Convert::ToString(splitAddrs, photons.size) << logger.end;
                logger.info << "Split left: " << Utils::CUDA::Convert::ToString(splitLeft, photons.size+1) << logger.end;
                logger.info << "===" << logger.end;
                logger.info << "Temp Children photon info: " << Utils::CUDA::Convert::ToString(tempChildren.photonInfo, activeRange * 2) << logger.end;
                logger.info << "Leaf side: " << Utils::CUDA::Convert::ToString(leafSide, activeRange * 2) << logger.end;
                logger.info << "" << logger.end;
                */
                cudaMemcpyFromSymbol(&createdLeafs, d_createdLeafs, sizeof(bool));

                if (createdLeafs){
                    // Sort leafs to the left
                    cudppScan(scanHandle, leafPrefix, leafSide, activeRange * 2);
                    /*
                    // Update upper node children with index and range,
                    // and update parent with real address.
                    logger.info << "=== MoveChildInfo<<<" << blocks << ", " << threads << ">>> ===" << logger.end;
                    logger.info << "activeIndex: " << activeIndex << logger.end;
                    logger.info << "activeRange: " << activeRange << logger.end;
                    logger.info << "Temp photon info: " << Utils::CUDA::Convert::ToString(tempChildren.photonInfo, activeRange * 2) << logger.end;
                    logger.info << "split side: " << Utils::CUDA::Convert::ToString(leafSide, activeRange * 2) << logger.end;
                    logger.info << "prefix sum: " << Utils::CUDA::Convert::ToString(leafPrefix, activeRange * 2) << logger.end;
                    */
                    MoveChildInfo<<<blocks, threads>>>(tempChildren.photonInfo,
                                                       leafSide, leafPrefix,
                                                       upperNodes.left + activeIndex, 
                                                       upperNodes.right + activeIndex,
                                                       upperNodes.photonInfo + activeIndex + activeRange,
                                                       upperNodes.info + activeIndex + activeRange);
                    /*
                    logger.info << "Child photon info: " << Utils::CUDA::Convert::ToString(upperNodes.photonInfo + activeIndex + activeRange, activeRange * 2) << logger.end;
                    logger.info << "Left: " << Utils::CUDA::Convert::ToString(upperNodes.left + activeIndex, activeRange) << logger.end;
                    logger.info << "Right: " << Utils::CUDA::Convert::ToString(upperNodes.right + activeIndex, activeRange) << logger.end;
                    */
                    CHECK_FOR_CUDA_ERROR();                    

                    cudaMemcpyFromSymbol(&leafsCreated, d_leafsCreated, sizeof(int));
                    //logger.info << "totalLeft: " << leafsCreated << logger.end;
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
                                                        upperNodes.right, upperNodes.photonInfo,
                                                        activePhotons);
                CHECK_FOR_CUDA_ERROR();

                /*
                logger.info << "=== UpdatePhotonOwners<<<" << blocks << ", " << threads << ">>> ===" << logger.end;
                logger.info << "owners: " << Utils::CUDA::Convert::ToString(photonOwners, activePhotons) << logger.end;
                logger.info << "left: " << Utils::CUDA::Convert::ToString(upperNodes.left, upperNodes.size) << logger.end;
                logger.info << "right: " << Utils::CUDA::Convert::ToString(upperNodes.right, upperNodes.size) << logger.end;
                logger.info << "photonInfo: " << Utils::CUDA::Convert::ToString(upperNodes.photonInfo, upperNodes.size) << logger.end;
                logger.info << "===" << logger.end;
                logger.info << "owners: " << Utils::CUDA::Convert::ToString(photonOwners, activePhotons) << logger.end;
                logger.info << "" << logger.end;
                */
            }
            
            void PhotonMap::PreprocessLowerNodes(){

                // Create lower nodes and their splitting planes.
                //cudaMemcpyToSymbol(d_upperPhotonInfo, &(upperNodes.photonInfo), sizeof(upperNodes.photonInfo));
                //cudaMemcpyToSymbol(d_lowerPhotonInfo, &(lowerNodes.photonInfo), sizeof(lowerNodes.photonInfo));
                //CHECK_FOR_CUDA_ERROR();

                if (lowerNodes.maxSize < upperNodeLeafList.size)
                    lowerNodes.Resize(upperNodeLeafList.size);

                unsigned int blocks, threads;
                Calc1DKernelDimensions(upperNodeLeafList.size, blocks, threads);
                CreateLowerNodes<<<blocks, threads>>>(upperNodeLeafList.leafIDs,
                                                      upperNodes.photonInfo,
                                                      lowerNodes.info,
                                                      lowerNodes.photonInfo,
                                                      upperNodeLeafList.size);

                lowerNodes.size = upperNodeLeafList.size;

                //logger.info << "Leaf IDs: " << Utils::CUDA::Convert::ToString(upperNodeLeafList.leafIDs, lowerNodes.size) << logger.end;
                //logger.info << "Lower node photon info: " << Utils::CUDA::Convert::ToString(lowerNodes.photonInfo, lowerNodes.size) << logger.end;

                
            }
            
            void PhotonMap::ProcessLowerNodes(int activeIndex,
                                              int activeRange,
                                              int &childrenCreated){

            }
            
        }
    }
}
