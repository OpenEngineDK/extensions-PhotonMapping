// Triangle map SAH creator interface for CUDA
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Utils/CUDA/TriangleMapSAHCreator.h>

#include <Scene/TriangleNode.h>
#include <Utils/CUDA/TriangleMap.h>
#include <Utils/CUDA/Utils.h>

#include <Utils/CUDA/Kernels/PhotonMapDeviceVars.h>

#include <Logging/Logger.h>

namespace OpenEngine {    
    using namespace Resources::CUDA;
    using namespace Scene;
    namespace Utils {
        namespace CUDA {

            using namespace Kernels;

            namespace TriangleMapSAHKernels {
#include <Utils/CUDA/Kernels/LowerTriangleMap.h>
            }
            using namespace TriangleMapSAHKernels;
            
            TriangleMapSAHCreator::TriangleMapSAHCreator() 
                : ITriangleMapCreator() {

                cutCreateTimer(&timerID);

                logger.info << "Create SAH Creator" << logger.end;

                splitTriangleSet =  new CUDADataBlock<1, int4>(1);
                childAreas = new CUDADataBlock<1, float2>(1);
                childSets = new CUDADataBlock<1, int2>(1);
                splitSide = new CUDADataBlock<1, int>(1);
                splitAddr = new CUDADataBlock<1, int>(1);

                scanConfig.op = CUDPP_ADD;
                scanConfig.datatype = CUDPP_INT;
                scanConfig.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
                scanSize = 262144;
                
                CUDPPResult res = cudppPlan(&scanHandle, scanConfig, scanSize, 1, 0);
                if (CUDPP_SUCCESS != res)
                    throw Core::Exception("Error creating CUDPP scanPlan");
            }
            
            TriangleMapSAHCreator::~TriangleMapSAHCreator() {
                if (splitTriangleSet) delete splitTriangleSet;
                if (childAreas) delete childAreas;
                if (childSets) delete childSets;
                if (splitSide) delete splitSide;
                if (splitAddr) delete splitAddr;
            }

            void TriangleMapSAHCreator::Create(TriangleMap* map,
                                               CUDADataBlock<1, int>* upperLeafIDs){
                
                primMin = map->primMin;
                primMax = map->primMax;

                int activeIndex = map->nodes->size; int activeRange = upperLeafIDs->GetSize();
                int childrenCreated;

                int triangles = map->primMin->GetSize();
                cudaMemcpyToSymbol(d_triangles, &triangles, sizeof(int));

                START_TIMER(timerID); 
                PreprocessLowerNodes(activeIndex, activeRange, map, upperLeafIDs);
                PRINT_TIMER(timerID, "Preprocess lower nodes using SAH");

                // @OPT Don't use a proxy, but do the first step outside the kernel.

                START_TIMER(timerID); 
                while (activeRange > 0){
                    ProcessLowerNodes(activeIndex, activeRange,
                                      map, childrenCreated);

                    activeIndex = map->nodes->size - childrenCreated;
                    activeRange = childrenCreated;
                }
                PRINT_TIMER(timerID, "Process lower nodes using SAH");
            }

            void TriangleMapSAHCreator::PreprocessLowerNodes(int activeIndex, int activeRange, 
                                      TriangleMap* map, Resources::CUDA::CUDADataBlock<1, int>* upperLeafIDs) {
                int triangles = primMin->GetSize();
                logger.info << "=== Preprocess " << activeRange << " Lower Nodes Starting at " << activeIndex << " === with " << triangles << " triangles" << logger.end;
                
                TriangleNode* nodes = map->nodes;
                nodes->Extend(activeIndex + activeRange);

                splitTriangleSet->Extend(triangles * 3);
                
                unsigned int blocks, threads;
                Calc1DKernelDimensions(activeRange, blocks, threads);
                PreprocesLowerNodes<<<blocks, threads>>>(upperLeafIDs->GetDeviceData(),
                                                         nodes->GetInfoData(),
                                                         nodes->GetPrimitiveInfoData(),
                                                         nodes->GetSurfaceAreaData(),
                                                         primMax->GetDeviceData(),
                                                         nodes->GetLeftData(), nodes->GetRightData(),
                                                         activeIndex, activeRange);
                CHECK_FOR_CUDA_ERROR();

                Calc1DKernelDimensions(activeRange * TriangleNode::MAX_LOWER_SIZE, blocks, threads, 448);
                unsigned int smemSize = threads * (sizeof(float3) + sizeof(float3));
                CreateSplittingPlanes<<<blocks, threads, smemSize>>>
                    (splitTriangleSet->GetDeviceData(), 
                     nodes->GetPrimitiveInfoData() + activeIndex,
                     primMin->GetDeviceData(), primMax->GetDeviceData(),
                     activeRange);
                CHECK_FOR_CUDA_ERROR();

#if CPU_VERIFY
                CheckPreprocess(activeIndex, activeRange, map, upperLeafIDs);
#endif

            }
                
            void TriangleMapSAHCreator::ProcessLowerNodes(int activeIndex, int activeRange, 
                                   TriangleMap* map, int &childrenCreated) {
                logger.info << "=== Process " << activeRange << " Lower Nodes Starting at " << activeIndex << " ===" << logger.end;

                TriangleNode* nodes = map->nodes;
                
                cudaMemcpyToSymbol(d_activeNodeIndex, &activeIndex, sizeof(int));
                cudaMemcpyToSymbol(d_activeNodeRange, &activeRange, sizeof(int));

                childAreas->Extend(activeRange);
                childSets->Extend(activeRange);
                splitSide->Extend(activeRange+1);
                splitAddr->Extend(activeRange+1);

                unsigned int blocks, threads;
                Calc1DKernelDimensions(activeRange, blocks, threads, 96);
                unsigned int smemSize = threads * TriangleNode::MAX_LOWER_SIZE * sizeof(float);
                //logger.info << "<<<" << blocks << ", " << threads << ", " << smemSize << ">>>" << logger.end;
                CalcSAH<<<blocks, threads, smemSize>>>(nodes->GetInfoData() + activeIndex,
                                                       nodes->GetSplitPositionData() + activeIndex,
                                                       nodes->GetPrimitiveInfoData() + activeIndex,
                                                       nodes->GetSurfaceAreaData() + activeIndex,
                                                       primMin->GetDeviceData(),
                                                       primMax->GetDeviceData(),
                                                       splitTriangleSet->GetDeviceData(),
                                                       childAreas->GetDeviceData(),
                                                       childSets->GetDeviceData(),
                                                       splitSide->GetDeviceData());
                CHECK_FOR_CUDA_ERROR();

                /*
                if (activeIndex == 2399){
                    logger.info << nodes->ToString(activeIndex+40) << logger.end;
                    
                    int primOffset = 632;
                    int primRange = 6;
                    
                    logger.info << "primMin: " << Convert::ToString(primMin->GetDeviceData() + primOffset, primRange) << logger.end;
                    logger.info << "primMax: " << Convert::ToString(primMax->GetDeviceData() + primOffset, primRange) << logger.end;
                    
                    logger.info << "X splittingSets: " << Convert::ToString(splitTriangleSet->GetDeviceData() + primOffset, primRange) << logger.end;
                    logger.info << "Y splittingSets: " << Convert::ToString(splitTriangleSet->GetDeviceData() + primOffset + triangles, primRange) << logger.end;
                    logger.info << "Z splittingSets: " << Convert::ToString(splitTriangleSet->GetDeviceData() + primOffset + 2 * triangles, primRange) << logger.end;
                    
                    logger.info << "Child areas: " << Convert::ToString(childAreas->GetDeviceData() + 40, 1) << logger.end;
                    logger.info << "Child sets: " << Convert::ToString(childSets->GetDeviceData() + 40, 1) << logger.end;
                }
                */

                cudppScan(scanHandle, splitAddr->GetDeviceData(), splitSide->GetDeviceData(), activeRange+1);
                CHECK_FOR_CUDA_ERROR();

                int splits;
                cudaMemcpy(&splits, splitAddr->GetDeviceData() + activeRange, sizeof(int), cudaMemcpyDeviceToHost);

                nodes->Extend(activeIndex + activeRange + 2 * splits);
                nodes->size = activeIndex + activeRange + 2 * splits;

                Calc1DKernelDimensions(activeRange, blocks, threads);
                CreateLowerSAHChildren<<<blocks, threads>>>(splitSide->GetDeviceData(),
                                                            splitAddr->GetDeviceData(),
                                                            childAreas->GetDeviceData(),
                                                            childSets->GetDeviceData(),
                                                            nodes->GetSurfaceAreaData(),
                                                            nodes->GetPrimitiveInfoData(),
                                                            splits);
                CHECK_FOR_CUDA_ERROR();

                childrenCreated = splits * 2;

            }

            void TriangleMapSAHCreator::CheckPreprocess(int activeIndex, int activeRange, 
                                 TriangleMap* map, Resources::CUDA::CUDADataBlock<1, int>* leafIDs) {

                TriangleNode* nodes = map->nodes;

                int h_leafIDs[activeRange];
                cudaMemcpy(h_leafIDs, leafIDs->GetDeviceData(), activeRange * sizeof(int), cudaMemcpyDeviceToHost);
                CHECK_FOR_CUDA_ERROR();

                char info[activeRange];
                int2 leafPrimInfo[activeRange];
                int left[activeRange];
                for (int i = 0; i < activeRange; ++i){
                    cudaMemcpy(info + i, nodes->GetInfoData() + h_leafIDs[i], sizeof(char), cudaMemcpyDeviceToHost);
                    cudaMemcpy(leafPrimInfo + i, nodes->GetPrimitiveInfoData() + h_leafIDs[i], sizeof(int2), cudaMemcpyDeviceToHost);
                    cudaMemcpy(left + i, nodes->GetLeftData() + h_leafIDs[i], sizeof(int), cudaMemcpyDeviceToHost);
                }
                CHECK_FOR_CUDA_ERROR();

                int2 lowerPrimInfo[activeRange];
                cudaMemcpy(lowerPrimInfo, nodes->GetPrimitiveInfoData() + activeIndex, activeRange * sizeof(int2), cudaMemcpyDeviceToHost);
                CHECK_FOR_CUDA_ERROR();

                for (int i = 0; i < activeRange; ++i){
                    if (info[i] == KDNode::LEAF){
                        if (lowerPrimInfo[i].y != 0)
                            throw Exception("Empty lower node didn't result in upper leaf.");
                    }else if (info[i] != KDNode::PROXY){
                        logger.info << nodes->ToString(h_leafIDs[i]) << logger.end;
                        logger.info << nodes->ToString(activeIndex + i) << logger.end;
                        throw Exception("info not equal to PROXY.");
                    }
                    if (left[i] != activeIndex + i)
                        throw Exception("leaf not pointing to correct lower node");
                }

                for (int i = 0; i < activeRange; ++i){
                    if (lowerPrimInfo[i].x != leafPrimInfo[i].x)
                        throw Exception("Leaf node " + Utils::Convert::ToString(h_leafIDs[i]) +
                                        "'s index " + Utils::Convert::ToString(leafPrimInfo[i].x) + 
                                        " does not match lower node " + Utils::Convert::ToString(activeIndex + i) +
                                        "'s " + Utils::Convert::ToString(lowerPrimInfo[i].x));
                    if (bitcount(lowerPrimInfo[i].y) > leafPrimInfo[i].y)
                        throw Exception("Leaf node " + Utils::Convert::ToString(h_leafIDs[i]) +
                                        "'s size of " + Utils::Convert::ToString(leafPrimInfo[i].y) + 
                                        " does not match lower node " + Utils::Convert::ToString(activeIndex + i) +
                                        "'s bitmap " + BitmapToString(lowerPrimInfo[i].y));
                }

                // @TODO check split set

            }
            
        }
    }
}
