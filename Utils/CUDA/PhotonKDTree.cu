// Photon KD tree class for CUDA
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Utils/CUDA/PhotonKDTree.h>

#include <Core/Exceptions.h>
#include <Math/RandomGenerator.h>

// Kernels
#include <Utils/CUDA/Kernels/ReduceBoundingBox.hcu>
#include <Utils/CUDA/Kernels/FinalBoundingBox.hcu>
#include <Utils/CUDA/Kernels/UpperNodeSplit.hcu>
#include <Utils/CUDA/Kernels/SortUpperChildren.hcu>

using namespace OpenEngine::Utils::CUDA::Kernels;

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {

            PhotonKDTree::PhotonKDTree(unsigned int size) {
                MAX_BLOCKS = (activeCudaDevice.maxGridSize[0] + 1) / activeCudaDevice.maxThreadsDim[0];

                logger.info << "MAX_BLOCKS " << MAX_BLOCKS << logger.end;

                // AABB calc vars
                aabbVars = AABBVar(MAX_BLOCKS);
                
                // Allocate photons on GPU
                photons = PhotonNode(size);

                // Calculate amount of nodes probably required
                unsigned int upperNodeSize = 2.5f * photons.maxSize / KDPhotonUpperNode::BUCKET_SIZE;
                upperNodes.Init(upperNodeSize);

                logger.info << "Photons " << size << logger.end;
                logger.info << "Upper nodes " << upperNodeSize << logger.end;

                // Split vars
                splitVars.Init(size);
                scanConfig.op = CUDPP_ADD;
                scanConfig.datatype = CUDPP_UINT;
                scanConfig.algorithm = CUDPP_SCAN;
                scanConfig.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;

                CUDPPResult res = cudppPlan(&scanHandle, scanConfig, size, 1, 0);

                if (CUDPP_SUCCESS != res)
                    throw Core::Exception("Error creating CUDPP scanPlan");

                // Photon bla values

                float3 hat[size];
                Math::RandomGenerator rand;
                for (unsigned int i = 0; i < size / 2; ++i)
                    hat[i] = make_float3(rand.UniformInt(0.0f, 5.0f),
                                         rand.UniformInt(0.0f, 10.0f),
                                         rand.UniformInt(0.0f, 10.0f));
                for (unsigned int i = size / 2; i < size; ++i)
                    hat[i] = make_float3(rand.UniformInt(5.0f, 10.0f),
                                         rand.UniformInt(0.0f, 10.0f),
                                         rand.UniformInt(0.0f, 10.0f));
                
                cudaMemcpy(photons.pos, hat, size * sizeof(float3), cudaMemcpyHostToDevice);
                photons.size = size;
            }

            void PhotonKDTree::Create() {
                unsigned int childrenCreated = 1, lowerCreated = 0;
                unsigned int activeIndex = 0, activeRange = 1;
                
                // Initialize kd root node data
                unsigned int i = 0;
                cudaMemcpy(upperNodes.photonIndex, &i, sizeof(unsigned int), cudaMemcpyHostToDevice);
                cudaMemcpy(upperNodes.range, &(photons.size), sizeof(unsigned int), cudaMemcpyHostToDevice);
                upperNodes.size = 1;

                // Create upper nodes
                unsigned int depth = 0, maxDepth = 3;
                while (childrenCreated != 0 && depth < maxDepth){
                    CreateUpperNodes(activeIndex, activeRange, 
                                     childrenCreated, lowerCreated);

                    for (unsigned int i = activeIndex; i < activeIndex + activeRange; ++i)
                        logger.info << upperNodes.ToString(i) << upperNodes.PhotonsToString(i, photons) << logger.end;

                    activeIndex += activeRange;
                    activeRange = childrenCreated;
                    depth++;

                }
                
                // Create lower nodes
            }

            void PhotonKDTree::CreateUpperNodes(unsigned int activeIndex, 
                                                unsigned int activeRange, 
                                                unsigned int &childrenCreated, 
                                                unsigned int &lowerCreated){
                
                ComputeBoundingBoxes(activeIndex, activeRange);
                CHECK_FOR_CUDA_ERROR();

                SplitUpperNodes(activeIndex, activeRange);
                CHECK_FOR_CUDA_ERROR();
                
                childrenCreated = SortChildren(activeIndex, activeRange);

                lowerCreated = activeRange - childrenCreated;
                upperNodes.size += activeRange * 2;
            }

            void PhotonKDTree::CreateLowerNodes(){

            }
            
            void PhotonKDTree::ComputeBoundingBoxes(unsigned int activeIndex,
                                                    unsigned int activeRange){

                logger.info << "Compute Bounding Boxes" << logger.end;

                unsigned int nodeRanges[activeRange];
                cudaMemcpy(nodeRanges, upperNodes.range + activeIndex, 
                           activeRange * sizeof(unsigned int), cudaMemcpyDeviceToHost);
                
                unsigned int nodeID = activeIndex;
                while (nodeID < activeIndex + activeRange){
                    unsigned int blocksUsed = 0;
                    do {
                        /*
                         * @TODO If the computations only require one
                         * block, then reduce them directly into the
                         * node arrays.
                         */

                        logger.info << "AABB for node " << nodeID;
            
                        unsigned int size = nodeRanges[nodeID - activeIndex];
                        logger.info << " of size " << size << logger.end;
                        unsigned int blocks, threads;
                        Calc1DKernelDimensions(size, blocks, threads);

                        int smemSize = (threads <= 32) ? 4 * threads * sizeof(float3) : 2 * threads * sizeof(float3);
            
                        // Execute kernel
                        switch(threads){
                        case 512:
                            ReduceBoundingBox<float3, 512><<< blocks, threads, smemSize >>>(photons, upperNodes, aabbVars, nodeID, blocksUsed);
                            break;
                        case 256:
                            ReduceBoundingBox<float3, 256><<< blocks, threads, smemSize >>>(photons, upperNodes, aabbVars, nodeID, blocksUsed);
                            break;
                        case 128:
                            ReduceBoundingBox<float3, 128><<< blocks, threads, smemSize >>>(photons, upperNodes, aabbVars, nodeID, blocksUsed);
                            break;
                        case 64:
                            ReduceBoundingBox<float3, 64><<< blocks, threads, smemSize >>>(photons, upperNodes, aabbVars, nodeID, blocksUsed);
                            break;
                        case 32:
                            ReduceBoundingBox<float3, 32><<< blocks, threads, smemSize >>>(photons, upperNodes, aabbVars, nodeID, blocksUsed);
                            break;
                        case 16:
                            ReduceBoundingBox<float3, 16><<< blocks, threads, smemSize >>>(photons, upperNodes, aabbVars, nodeID, blocksUsed);
                            break;
                        case 8:
                            ReduceBoundingBox<float3, 8><<< blocks, threads, smemSize >>>(photons, upperNodes, aabbVars, nodeID, blocksUsed);
                            break;
                        case 4:
                            ReduceBoundingBox<float3, 4><<< blocks, threads, smemSize >>>(photons, upperNodes, aabbVars, nodeID, blocksUsed);
                            break;
                        case 2:
                            ReduceBoundingBox<float3, 2><<< blocks, threads, smemSize >>>(photons, upperNodes, aabbVars, nodeID, blocksUsed);
                            break;
                        }
                        CHECK_FOR_CUDA_ERROR();

                        blocksUsed += blocks;
                        nodeID++;
                    }while(blocksUsed < MAX_BLOCKS && nodeID < activeIndex + activeRange);

                    // Run segmented reduce ie FinalBoundingBox
                    unsigned int iterations = log2(blocksUsed * 1.0f);
                    //cutResetTimer(timerID);
                    //cutStartTimer(timerID);
                    FinalBoundingBox1<<<1, blocksUsed>>>(aabbVars, upperNodes, iterations);
                    CHECK_FOR_CUDA_ERROR();
                    /*
                      cudaThreadSynchronize();
                      CHECK_FOR_CUDA_ERROR();
                      cutStopTimer(timerID);
                    */    

                    //logger.info << "Final bounding box time: " << cutGetTimerValue(timerID) << "ms" << logger.end;
                }
            }
            
            void PhotonKDTree::SplitUpperNodes(unsigned int activeIndex,
                                               unsigned int activeRange){

                logger.info << "Split Upper Nodes" << logger.end;

                SetUpperNodeSplitPlane<<< 1, activeRange>>>(upperNodes, activeIndex);
                CHECK_FOR_CUDA_ERROR();

                // Split each node along the spatial median
                char axes[activeRange];
                cudaMemcpy(axes, upperNodes.info+activeIndex, 
                           activeRange * sizeof(char), cudaMemcpyDeviceToHost);
                CHECK_FOR_CUDA_ERROR();
                
                unsigned int photonIndices[activeRange];
                cudaMemcpy(photonIndices, upperNodes.photonIndex + activeIndex, 
                           activeRange * sizeof(unsigned int), cudaMemcpyDeviceToHost);
                CHECK_FOR_CUDA_ERROR();
                
                unsigned int photonRanges[activeRange];
                cudaMemcpy(photonRanges, upperNodes.range+activeIndex, 
                           activeRange * sizeof(unsigned int), cudaMemcpyDeviceToHost);
                CHECK_FOR_CUDA_ERROR();
                    
                for (unsigned int nodeID = 0; nodeID < activeRange; ++nodeID){

                    unsigned int blocks, threads;
                    Calc1DKernelDimensions(photonRanges[nodeID], blocks, threads);
        
                    // For each position calculate the side it should
                    // be on after the split.
                    switch (axes[nodeID]) {
                    case KDPhotonUpperNode::X:
                        CalcSplitSide<KDPhotonUpperNode::X><<<blocks, threads>>>(splitVars, upperNodes, photons, photonRanges[nodeID], nodeID + activeIndex);
                        break;
                    case KDPhotonUpperNode::Y:
                        CalcSplitSide<KDPhotonUpperNode::Y><<<blocks, threads>>>(splitVars, upperNodes, photons, photonRanges[nodeID], nodeID + activeIndex);
                        break;
                    case KDPhotonUpperNode::Z:
                        CalcSplitSide<KDPhotonUpperNode::Z><<<blocks, threads>>>(splitVars, upperNodes, photons, photonRanges[nodeID], nodeID + activeIndex);
                        break;
                    }
                    CHECK_FOR_CUDA_ERROR();

                    // Scan the array of split sides and calculate
                    // it's prefix sum.
                    cudppScan(scanHandle, splitVars.prefixSum,
                              splitVars.side,
                              photonRanges[nodeID]);
                    CHECK_FOR_CUDA_ERROR();


                    // Move the photon positions and it's association indices.
                    SplitPhotons<<<blocks, threads>>>(splitVars, photons, upperNodes, nodeID + activeIndex);
                    CHECK_FOR_CUDA_ERROR();
                    
                    // Only copy moved photons or copy the entire array?
                    // Since almost it will almost always be the entire
                    // array that is modifed we can start by copying that.
                    // Perhaps only do the memcpy when the previous
                    // photons and currents photons aren't 'neightbours'.
                    
                    cudaMemcpy(photons.pos + photonIndices[nodeID], 
                               splitVars.tempPos + photonIndices[nodeID], 
                               photonRanges[nodeID] * sizeof(float3), 
                               cudaMemcpyDeviceToDevice);
                }
            }

            unsigned int PhotonKDTree::SortChildren(unsigned int activeIndex,
                                                    unsigned int activeRange){

                logger.info << "Sort children" << logger.end;

                unsigned int blocks, threads;
                Calc1DKernelDimensions(activeRange * 2, blocks, threads);

                bool h_hasLeafs = false;
                cudaMemcpyToSymbol(d_hasLeafs, &h_hasLeafs, sizeof(bool));
                CHECK_FOR_CUDA_ERROR();
                
                MarkUpperLeafs<<<blocks, threads>>>(splitVars, upperNodes, 
                                                    activeIndex, activeRange);
                CHECK_FOR_CUDA_ERROR();
                
                cudaMemcpyFromSymbol(&h_hasLeafs, d_hasLeafs, sizeof(bool));
                CHECK_FOR_CUDA_ERROR();
                
                if (h_hasLeafs){
                    // One or more nodes are leafs and should be sorted

                    // Scan the array of split sides and calculate
                    // it's prefix sum.
                    cudppScan(scanHandle, splitVars.prefixSum,
                              splitVars.side,
                              activeRange);
                    CHECK_FOR_CUDA_ERROR();

                    //SplitUpperLeafs<<<blocks, threads>>>();
                    
                    return 0;
                }else{
                    // Else none of the nodes are leafs and the active
                    // range doubles.
                    return activeRange * 2;
                }
            }
            

        }
    }
}
