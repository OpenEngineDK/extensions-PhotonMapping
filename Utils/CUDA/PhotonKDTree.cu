// Photon KD tree class for CUDA
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Utils/CUDA/PhotonKDTree.h>
#include <Math/RandomGenerator.h>

#include <META/CUDPP.h>

// Kernels
#include <Utils/CUDA/Kernels/ReduceBoundingBox.hcu>
#include <Utils/CUDA/Kernels/FinalBoundingBox.hcu>
#include <Utils/CUDA/Kernels/UpperNodeSplit.hcu>

using namespace OpenEngine::Utils::CUDA::Kernels;

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {

            PhotonKDTree::PhotonKDTree(unsigned int size) {
                // AABB calc vars
                aabbVars = AABBVar(MAX_BLOCKS);
                
                // Allocate photons on GPU
                photons = PhotonNode(size);

                // Calculate amount of nodes probably required
                unsigned int upperNodeSize = 2.5f * photons.maxSize / BUCKET_SIZE;
                upperNodes.Init(upperNodeSize);

                logger.info << "Photons " << size << logger.end;
                logger.info << "Upper nodes " << upperNodeSize << logger.end;

                splitVars.Init(size);

                float3 hat[size];
                Math::RandomGenerator rand;
                for (unsigned int i = 0; i < size / 2; ++i)
                    hat[i] = make_float3(rand.UniformFloat(0.0f, 5.0f),
                                         rand.UniformFloat(0.0f, 10.0f),
                                         rand.UniformFloat(0.0f, 10.0f));
                for (unsigned int i = size / 2; i < size; ++i)
                    hat[i] = make_float3(rand.UniformFloat(5.0f, 10.0f),
                                         rand.UniformFloat(0.0f, 10.0f),
                                         rand.UniformFloat(0.0f, 10.0f));
                
                cudaMemcpy(photons.pos, hat, size * sizeof(float3), cudaMemcpyHostToDevice);
                photons.size = size;
            }

            void PhotonKDTree::Create() {
                unsigned int childrenCreated = 1, lowerCreated = 0;
                unsigned int activeIndex = 0, activeRange = 1;
                
                // Initialize kd root node data
                unsigned int i = 0;
                cudaMemcpy(upperNodes.startIndex, &i, sizeof(unsigned int), cudaMemcpyHostToDevice);
                cudaMemcpy(upperNodes.range, &(photons.size), sizeof(unsigned int), cudaMemcpyHostToDevice);
                upperNodes.size = 1;

                // Create upper nodes
                while (childrenCreated != 0){
                    CreateUpperNodes(activeIndex, activeRange, 
                                     childrenCreated, lowerCreated);

                    activeIndex += activeRange;
                    activeRange = childrenCreated;
                }
                
                // Print root node
                logger.info << upperNodes.ToString(0) << logger.end;
    
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
                
                childrenCreated = 0;
                lowerCreated = 0;
            }

            void PhotonKDTree::CreateLowerNodes(){

            }
            
            void PhotonKDTree::ComputeBoundingBoxes(unsigned int activeIndex,
                                                    unsigned int activeRange){

                logger.info << "Compute Bounding Box" << logger.end;

                unsigned int nodeRanges[activeRange];
                cudaMemcpy(nodeRanges, upperNodes.range + activeIndex, 
                           activeRange * sizeof(unsigned int), cudaMemcpyDeviceToHost);
                
                unsigned int nodeID = activeIndex;
                while (nodeID < activeIndex + activeRange){
                    unsigned int blocksUsed = 0;
                    do {
                        /*
                         * @TODO Extend to compute multiple bounding boxes instead of
                         * computing one at a time.
                         */

                        logger.info << "AABB for node " << nodeID << logger.end;
            
                        unsigned int size = nodeRanges[nodeID - activeIndex];
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
                    unsigned int iterations = log2(blocksUsed * 1.0f) - 1;
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
                cudaMemcpy(axes, upperNodes.info+activeIndex, activeRange * sizeof(char), cudaMemcpyDeviceToHost);
                CHECK_FOR_CUDA_ERROR();
                
                unsigned int photonRanges[activeRange];
                cudaMemcpy(photonRanges, upperNodes.range+activeIndex, activeRange * sizeof(unsigned int), cudaMemcpyDeviceToHost);
                CHECK_FOR_CUDA_ERROR();
                    
                for (unsigned int nodeID = 0; nodeID < activeRange; ++nodeID){

                    unsigned int blocks, threads;
                    Calc1DKernelDimensions(photonRanges[nodeID], blocks, threads);
        
                    // For each position calculate the side it should
                    // be on after the split.
                    switch (axes[nodeID]) {
                    case KDPhotonUpperNode::X:
                        CalcSplitSide<KDPhotonUpperNode::X><<<blocks, threads>>>(splitVars, upperNodes, photons, photonRanges[nodeID], nodeID + activeIndex);
                    case KDPhotonUpperNode::Y:
                        CalcSplitSide<KDPhotonUpperNode::Y><<<blocks, threads>>>(splitVars, upperNodes, photons, photonRanges[nodeID], nodeID + activeIndex);
                    case KDPhotonUpperNode::Z:
                        CalcSplitSide<KDPhotonUpperNode::Z><<<blocks, threads>>>(splitVars, upperNodes, photons, photonRanges[nodeID], nodeID + activeIndex);
                    }
                    CHECK_FOR_CUDA_ERROR();

                    unsigned int photonIndex;
                    cudaMemcpy(&photonIndex, upperNodes.startIndex + activeIndex, sizeof(unsigned int), cudaMemcpyDeviceToHost);
                    CHECK_FOR_CUDA_ERROR();
                    /*
                      logger.info << "Photons " << photons.PositionToString(photonIndex, photonRanges[nodeID]) << logger.end;
                      logger.info << "Sides " << splitVars.SideToString(photonIndex, photonRanges[nodeID]) << logger.end;
                    */

                    // Scan the array of split sides and calculate
                    // it's prefix sum.

                    // Move the photons and it's association indices.

                }
            }

        }
    }
}
