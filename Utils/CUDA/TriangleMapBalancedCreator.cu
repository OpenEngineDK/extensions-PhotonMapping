// Triangle map balanced creator interface for CUDA
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Utils/CUDA/TriangleMapBalancedCreator.h>

#include <Scene/TriangleNode.h>
#include <Utils/CUDA/Convert.h>
#include <Utils/CUDA/TriangleMap.h>
#include <Utils/CUDA/Utils.h>
#include <Logging/Logger.h>

#include <Utils/CUDA/Kernels/PhotonMapDeviceVars.h>

namespace OpenEngine {    
    using namespace Resources::CUDA;
    using namespace Scene;
    namespace Utils {
        namespace CUDA {

            using namespace Kernels;

            namespace TriangleMapBalancedKernels {
#include <Utils/CUDA/Kernels/LowerTriangleMap.h>
            }
            using namespace TriangleMapBalancedKernels;

            TriangleMapBalancedCreator::TriangleMapBalancedCreator()
                : ITriangleMapCreator() {

                cutCreateTimer(&timerID);

                logger.info << "Create balanced lower tree creator" << logger.end;

                splitTriangleSet =  new CUDADataBlock<1, int4>(1);
                childSets = new CUDADataBlock<1, int2>(1);
                splitSide = new CUDADataBlock<1, int>(1);
                splitAddr = new CUDADataBlock<1, int>(1);

                scanConfig.algorithm = CUDPP_SCAN;
                scanConfig.op = CUDPP_ADD;
                scanConfig.datatype = CUDPP_INT;
                scanConfig.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
                scanSize = 262144;
                
                CUDPPResult res = cudppPlan(&scanHandle, scanConfig, scanSize, 1, 0);
                if (CUDPP_SUCCESS != res)
                    throw Core::Exception("Error creating CUDPP scanPlan for Triangle Map SAH Creator");
            }

            TriangleMapBalancedCreator::~TriangleMapBalancedCreator(){
                if (splitTriangleSet) delete splitTriangleSet;
                if (childSets) delete childSets;
                if (splitSide) delete splitSide;
                if (splitAddr) delete splitAddr;
            }
            
            void TriangleMapBalancedCreator::Create(TriangleMap* map, 
                                                    CUDADataBlock<1, int>* upperLeafIDs){

                primMin = map->primMin;
                primMax = map->primMax;

                int activeIndex = map->nodes->GetSize(); int activeRange = upperLeafIDs->GetSize();
                int childrenCreated;

                int triangles = map->primMin->GetSize();
                cudaMemcpyToSymbol(d_triangles, &triangles, sizeof(int));

                START_TIMER(timerID); 
                PreprocessLowerNodes(activeIndex, activeRange, map, upperLeafIDs);
                PRINT_TIMER(timerID, "Preprocess lower nodes");

                START_TIMER(timerID); 
                ProcessLowerNodes(activeIndex, activeRange,
                                  map, upperLeafIDs, childrenCreated);
                
                activeIndex = map->nodes->GetSize() - childrenCreated;
                activeRange = childrenCreated;

                while (activeRange > 0){
                    ProcessLowerNodes(activeIndex, activeRange,
                                      map, NULL, childrenCreated);

                    activeIndex = map->nodes->GetSize() - childrenCreated;
                    activeRange = childrenCreated;
                }
                PRINT_TIMER(timerID, "Process lower nodes into balanced subtrees");
            }
            
            void TriangleMapBalancedCreator::PreprocessLowerNodes(int activeIndex, int activeRange, 
                                                                  TriangleMap* map, CUDADataBlock<1, int>* upperLeafIDs){
                int triangles = primMin->GetSize();
                logger.info << "=== Preprocess " << activeRange << " Lower Nodes Starting at " << activeIndex << " === with " << triangles << " indices" << logger.end;
                
                TriangleNode* nodes = map->nodes;

                splitTriangleSet->Extend(triangles * 3);
                
                unsigned int blocks, threads, smemSize;
                Calc1DKernelDimensions(activeRange, blocks, threads);
                PreprocessLeafNodes<<<blocks, threads>>>(upperLeafIDs->GetDeviceData(),
                                                         nodes->GetPrimitiveInfoData(),
                                                         activeRange);
                CHECK_FOR_CUDA_ERROR();
                
                unsigned int smemPrThread = sizeof(float3) + sizeof(float3);
                Calc1DKernelDimensionsWithSmem(activeRange * TriangleNode::MAX_LOWER_SIZE, smemPrThread, 
                                               blocks, threads, smemSize, 448);
                CreateSplittingPlanes<<<blocks, threads, smemSize>>>
                    (upperLeafIDs->GetDeviceData(),
                     nodes->GetPrimitiveInfoData(),
                     primMin->GetDeviceData(), primMax->GetDeviceData(),
                     splitTriangleSet->GetDeviceData(), 
                     activeIndex, activeRange);
                CHECK_FOR_CUDA_ERROR();
            }
            
            void TriangleMapBalancedCreator::ProcessLowerNodes(int activeIndex, int activeRange, 
                                                               TriangleMap* map, CUDADataBlock<1, int>* upperLeafIDs,
                                                               int &childrenCreated){
                logger.info << "=== Process " << activeRange << " Lower Nodes Starting at " << activeIndex << " ===" << logger.end;

                TriangleNode* nodes = map->nodes;

                cudaMemcpyToSymbol(d_activeNodeIndex, &activeIndex, sizeof(int));
                cudaMemcpyToSymbol(d_activeNodeRange, &activeRange, sizeof(int));

                childSets->Extend(activeRange);
                splitSide->Extend(activeRange+1);
                splitAddr->Extend(activeRange+1);

                unsigned int blocks, threads, smemSize;
                unsigned int smemPrThread = TriangleNode::MAX_LOWER_SIZE * sizeof(float);
                Calc1DKernelDimensionsWithSmem(activeRange, smemPrThread, 
                                               blocks, threads, smemSize, 96);

                if (upperLeafIDs)
                    CalcSplit<true><<<blocks, threads>>>(upperLeafIDs->GetDeviceData(), 
                                                         nodes->GetInfoData(),
                                                         nodes->GetSplitPositionData(),
                                                         nodes->GetPrimitiveInfoData(),
                                                         primMin->GetDeviceData(),
                                                         primMax->GetDeviceData(),
                                                         splitTriangleSet->GetDeviceData(),
                                                         childSets->GetDeviceData(),
                                                         splitSide->GetDeviceData());
                else
                    CalcSplit<false><<<blocks, threads>>>(NULL, nodes->GetInfoData(),
                                                          nodes->GetSplitPositionData(),
                                                          nodes->GetPrimitiveInfoData(),
                                                          primMin->GetDeviceData(),
                                                          primMax->GetDeviceData(),
                                                          splitTriangleSet->GetDeviceData(),
                                                          childSets->GetDeviceData(),
                                                          splitSide->GetDeviceData());
                CHECK_FOR_CUDA_ERROR();

                cudppScan(scanHandle, splitAddr->GetDeviceData(), splitSide->GetDeviceData(), activeRange+1);
                CHECK_FOR_CUDA_ERROR();

                int splits;
                cudaMemcpy(&splits, splitAddr->GetDeviceData() + activeRange, sizeof(int), cudaMemcpyDeviceToHost);
                nodes->Extend(activeIndex + activeRange + 2 * splits);

                Calc1DKernelDimensions(activeRange, blocks, threads);
                if (upperLeafIDs)
                    CreateChildren<true><<<blocks, threads>>>(upperLeafIDs->GetDeviceData(), 
                                                              splitSide->GetDeviceData(),
                                                              splitAddr->GetDeviceData(),
                                                              childSets->GetDeviceData(),
                                                              nodes->GetPrimitiveInfoData(),
                                                              nodes->GetChildrenData(),
                                                              splits);
                else
                    CreateChildren<false><<<blocks, threads>>>(NULL, splitSide->GetDeviceData(),
                                                               splitAddr->GetDeviceData(),
                                                               childSets->GetDeviceData(),
                                                               nodes->GetPrimitiveInfoData(),
                                                               nodes->GetChildrenData(),
                                                               splits);
                CHECK_FOR_CUDA_ERROR();

                childrenCreated = splits * 2;
            }

        }
    }
}