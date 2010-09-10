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
#include <Utils/CUDA/Kernels/UpperNodeSplit.hcu>
#include <Utils/CUDA/Kernels/UpperNodeChildren.hcu>

using namespace OpenEngine::Utils::CUDA::Kernels;

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {

            PhotonKDTree::PhotonKDTree(unsigned int size) {
                MAX_BLOCKS = (activeCudaDevice.maxGridSize[0] + 1) / activeCudaDevice.maxThreadsDim[0];

                logger.info << "MAX_BLOCKS " << MAX_BLOCKS << logger.end;
                
                // Initialized timer
                cutCreateTimer(&timerID);

                // AABB calc vars
                aabbVars = AABBVar(MAX_BLOCKS);
                
                // Allocate photons on GPU
                photons = PhotonNode(size);
                photons.CreateRandomData();

                // Calculate amount of nodes probably required
                //unsigned int upperNodeSize = 2.5f * photons.maxSize / KDPhotonUpperNode::BUCKET_SIZE;
                unsigned int upperNodeSize = 1;
                upperNodes = KDPhotonUpperNode(upperNodeSize);

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
                while (childrenCreated != 0){
                    CreateUpperNodes(activeIndex, activeRange, 
                                     childrenCreated, lowerCreated);
                    /*
                    for (unsigned int i = activeIndex; i < activeIndex + activeRange; ++i)
                        logger.info << upperNodes.ToString(i) << upperNodes.PhotonsToString(i, photons) << logger.end;
                    */
                    activeIndex += activeRange;
                    activeRange = childrenCreated;
                }
                
                // Create lower nodes
                CreateLowerNodes();
            }

            void PhotonKDTree::CreateUpperNodes(unsigned int activeIndex, 
                                                unsigned int activeRange, 
                                                unsigned int &childrenCreated, 
                                                unsigned int &lowerCreated){

                // Check that there is room for the new children
                if (upperNodes.maxSize < upperNodes.size + activeRange * 2)
                    upperNodes.Resize(upperNodes.size + activeRange * 2);

                // Copy data needed for cpu bookkeeping.
                unsigned int photonRanges[activeRange];
                cudaMemcpy(photonRanges, upperNodes.range + activeIndex, 
                           activeRange * sizeof(unsigned int), cudaMemcpyDeviceToHost);

                // Compute the bounding box of upper nodes.
                // Leaf nodes possibly excepted.
                ComputeBoundingBoxes(activeIndex, activeRange, photonRanges);
                CHECK_FOR_CUDA_ERROR();

                // Create links to children
                childrenCreated = CreateChildren(activeIndex, activeRange);
                CHECK_FOR_CUDA_ERROR();

                SplitUpperNodePhotons(activeIndex, activeRange, photonRanges);
                CHECK_FOR_CUDA_ERROR();
                
                lowerCreated = activeRange * 2 - childrenCreated;
                upperNodes.size += childrenCreated;
            }

            void PhotonKDTree::ComputeBoundingBoxes(unsigned int activeIndex,
                                                    unsigned int activeRange,
                                                    unsigned int *photonRanges){
                
                logger.info << "Compute bounding boxes" << logger.end;

                /*
                cudaMemcpyToSymbol(ReduceBoundingBoxNS::photonPos, photons.pos, 
                                   sizeof(point*), 0, cudaMemcpyDeviceToDevice);
                CHECK_FOR_CUDA_ERROR();
                */
                
                unsigned int blocksUsed = 0;
                for (unsigned int nodeID = activeIndex; 
                     nodeID < activeIndex + activeRange;
                     ++nodeID){

                    /*
                     * @TODO If the computations only require one
                     * block, then reduce them directly into the
                     * node arrays.
                     * -- or --
                     * Reduce all nodes only requiring 1 block at
                     * once. Easily to manage and will give a
                     * performance upgrade on ALL small nodes (of
                     * which there are many)
                     *
                     * Skip AABB calculations of too small
                     * nodes. No need to calc it for the
                     * leafs. (Is there?)
                     */
                    
                    unsigned int size = photonRanges[nodeID - activeIndex];
                    unsigned int blocks, threads;
                    Calc1DKernelDimensions(size, blocks, threads);
                    int smemSize = (threads <= 32) ? 4 * threads * sizeof(point) : 2 * threads * sizeof(point);

                    /**
                     * If the next reduce will use to many blocks,
                     * then do a segmented reduce and reset blocks
                     * used count.
                     */
                    if (blocksUsed + blocks > MAX_BLOCKS){
                        SegmentedReduce(blocksUsed);
                        blocksUsed = 0;
                    }
                    
                    cudaMemcpyToSymbol(ReduceBoundingBoxNS::photonIndex, upperNodes.photonIndex + nodeID, sizeof(unsigned int), 0, cudaMemcpyDeviceToDevice);
                    cudaMemcpyToSymbol(ReduceBoundingBoxNS::photonRange, upperNodes.range + nodeID, sizeof(unsigned int), 0, cudaMemcpyDeviceToDevice);
                    CHECK_FOR_CUDA_ERROR();
                    
                    // Execute kernel
                    switch(threads){
                    case 512:
                        ReduceBoundingBox<512><<< blocks, threads, smemSize >>>(photons.pos, aabbVars, nodeID, blocksUsed);
                        break;
                    case 256:
                        ReduceBoundingBox<256><<< blocks, threads, smemSize >>>(photons.pos, aabbVars, nodeID, blocksUsed);
                        break;
                    case 128:
                        ReduceBoundingBox<128><<< blocks, threads, smemSize >>>(photons.pos, aabbVars, nodeID, blocksUsed);
                        break;
                    case 64:
                        ReduceBoundingBox<64><<< blocks, threads, smemSize >>>(photons.pos, aabbVars, nodeID, blocksUsed);
                        break;
                    case 32:
                        ReduceBoundingBox<32><<< blocks, threads, smemSize >>>(photons.pos, aabbVars, nodeID, blocksUsed);
                        break;
                    case 16:
                        ReduceBoundingBox<16><<< blocks, threads, smemSize >>>(photons.pos, aabbVars, nodeID, blocksUsed);
                        break;
                    case 8:
                        ReduceBoundingBox<8><<< blocks, threads, smemSize >>>(photons.pos, aabbVars, nodeID, blocksUsed);
                        break;
                    case 4:
                        ReduceBoundingBox<4><<< blocks, threads, smemSize >>>(photons.pos, aabbVars, nodeID, blocksUsed);
                        break;
                    case 2:
                        ReduceBoundingBox<2><<< blocks, threads, smemSize >>>(photons.pos, aabbVars, nodeID, blocksUsed);
                        break;
                    case 1:
                        ReduceBoundingBox<1><<< blocks, threads, smemSize >>>(photons.pos, aabbVars, nodeID, blocksUsed);
                        break;
                    }
                    CHECK_FOR_CUDA_ERROR();

                    blocksUsed += blocks;
                }
                
                // Run segmented reduce ie FinalBoundingBox
                if (blocksUsed > 0){
                    SegmentedReduce(blocksUsed);
                }
            }

            void PhotonKDTree::SegmentedReduce(unsigned int blocksUsed){
                // Setup device variables
                cudaMemcpyToSymbol(ReduceBoundingBoxNS::startNode, aabbVars.owner, 
                                   sizeof(unsigned int), 0, cudaMemcpyDeviceToDevice);
                cudaMemcpyToSymbol(ReduceBoundingBoxNS::endNode, aabbVars.owner + blocksUsed - 1, 
                                   sizeof(unsigned int), 0, cudaMemcpyDeviceToDevice);
                CHECK_FOR_CUDA_ERROR();
                
                //unsigned int iterations = log2(blocksUsed * 1.0f);
                        
                //cutResetTimer(timerID);
                //cutStartTimer(timerID);
                if (blocksUsed <= 128)
                    FinalBoundingBox3<128><<<1, blocksUsed>>>(aabbVars, upperNodes);
                else
                    throw Core::Exception("used more blocks than 128, how the hell?");
                CHECK_FOR_CUDA_ERROR();
                /*
                  cudaThreadSynchronize();
                  CHECK_FOR_CUDA_ERROR();
                  cutStopTimer(timerID);
                  logger.info << "Final bounding box 3 time: " << cutGetTimerValue(timerID) << "ms" << logger.end;
                */
            }
            
            unsigned int PhotonKDTree::CreateChildren(unsigned int activeIndex,
                                                      unsigned int activeRange){

                logger.info << "Sort " << activeRange << " children starting from node " << activeIndex << logger.end;

                unsigned int blocks, threads;
                Calc1DKernelDimensions(activeRange, blocks, threads);

                unsigned int children = activeRange;

                bool h_onlyChildren = true;
                cudaMemcpyToSymbol(d_onlyChildren, &h_onlyChildren, sizeof(bool));
                CHECK_FOR_CUDA_ERROR();
                
                MarkUpperLeafs<<<blocks, threads>>>(splitVars, upperNodes, 
                                                    activeIndex, activeRange);
                CHECK_FOR_CUDA_ERROR();
                
                cudaMemcpyFromSymbol(&h_onlyChildren, d_onlyChildren, sizeof(bool));
                CHECK_FOR_CUDA_ERROR();

                if (!h_onlyChildren){
                    // Create children for the nodes that require them

                    // Scan the array of split sides and calculate
                    // it's prefix sum.
                    cudppScan(scanHandle, splitVars.prefixSum,
                              splitVars.side,
                              activeRange);
                    CHECK_FOR_CUDA_ERROR();

                    SetupSomeChildLinks<<<blocks, threads>>>(splitVars, upperNodes, activeIndex, activeRange);
                    CHECK_FOR_CUDA_ERROR();

                    cudaMemcpyFromSymbol(&children, newChildren, sizeof(unsigned int));
                    
                }else{
                    // Setup children for all nodes, since none are leafs.
                    SetupChildLinks<<<blocks, threads>>>(upperNodes.child,
                                                         upperNodes.parent,
                                                         upperNodes.size,
                                                         activeIndex,
                                                         activeRange);
                    CHECK_FOR_CUDA_ERROR();
                }

                return children * 2;
            }

            void PhotonKDTree::SplitUpperNodePhotons(unsigned int activeIndex,
                                                     unsigned int activeRange,
                                                     unsigned int *photonRanges){

                logger.info << "Split Upper Nodes" << logger.end;

                /**
                 * @TODO
                 *
                 * Axes need not be copied if the kernel branches over
                 * the sides. No thread would diverge. Maybe then the
                 * calculations could be done for all nodes in
                 * parallel (would cause diverging)
                 *
                 * Set Upper Node split plane right after bbox
                 * calcs. Then do asynchrunous indices and axis
                 * copying.
                 */

                unsigned int blocks, threads;
                Calc1DKernelDimensions(activeRange, blocks, threads);

                SetUpperNodeSplitPlane<<< blocks, threads>>>(upperNodes, activeIndex, activeRange);
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
                
                for (unsigned int nodeID = 0; nodeID < activeRange; ++nodeID){
                    if (photonRanges[nodeID] > KDPhotonUpperNode::BUCKET_SIZE){
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
                        // Also set the childrens photon range and index
                        SplitPhotons<<<blocks, threads>>>(splitVars, photons, upperNodes, nodeID + activeIndex);
                        CHECK_FOR_CUDA_ERROR();
                        
                        // @TODO extend to copy larger collective chunks
                        // instead of doing a memcpy every god damn time.
                        
                        // Or perhaps launch a kernel that can do the
                        // memcpy without knowing the damn photon indices?
                        
                        cudaMemcpy(photons.pos + photonIndices[nodeID], 
                                   splitVars.tempPos + photonIndices[nodeID], 
                                   photonRanges[nodeID] * sizeof(point), 
                                   cudaMemcpyDeviceToDevice);
                    }
                }
            }
            
            void PhotonKDTree::CreateLowerNodes(){

            }
            

        }
    }
}
