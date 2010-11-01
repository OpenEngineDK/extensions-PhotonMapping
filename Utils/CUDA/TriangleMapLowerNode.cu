// Triangle map class for lower nodes
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Utils/CUDA/TriangleMap.h>
#include <Utils/CUDA/Utils.h>
#include <Utils/CUDA/Convert.h>

#include <Utils/CUDA/Kernels/LowerTriangleMap.h>

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {

            using namespace Kernels;

            void TriangleMap::CreateLowerNodes(){
                int activeIndex = nodes->size; int activeRange = upperNodeLeafs;
                int childrenCreated;

                cudaMemcpyToSymbol(d_triangles, &triangles, sizeof(int));

                START_TIMER(timerID); 
                PreprocessLowerNodes(activeIndex, activeRange);
                PRINT_TIMER(timerID, "Preprocess lower nodes");

                START_TIMER(timerID); 
                while (activeRange > 0){
                    ProcessLowerNodes(activeIndex, activeRange,
                                      childrenCreated);

                    activeIndex = nodes->size - childrenCreated;
                    activeRange = childrenCreated;
                }
                PRINT_TIMER(timerID, "triangle lower map");
            }

            void TriangleMap::PreprocessLowerNodes(int activeIndex, int activeRange){
                logger.info << "=== Preprocess " << activeRange << " Lower Nodes Starting at " << activeIndex << " === with " << triangles << " triangles" << logger.end;
                
                nodes->Extend(nodes->size + activeRange);

                splitTriangleSet->Extend(triangles * 3);
                
                unsigned int blocks, threads;
                Calc1DKernelDimensions(activeRange, blocks, threads);
                PreprocesLowerNodes<<<blocks, threads>>>(upperNodeLeafList->GetDeviceData(),
                                                         nodes->GetInfoData(),
                                                         nodes->GetPrimitiveInfoData(),
                                                         nodes->GetSurfaceAreaData(),
                                                         resultMax->GetDeviceData(),
                                                         nodes->GetLeftData(), nodes->GetRightData(),
                                                         activeIndex, activeRange);
                CHECK_FOR_CUDA_ERROR();

                Calc1DKernelDimensions(activeRange * TriangleNode::MAX_LOWER_SIZE, blocks, threads, 448);
                unsigned int smemSize = threads * (sizeof(float3) + sizeof(float3));
                CreateSplittingPlanes<<<blocks, threads, smemSize>>>
                    (splitTriangleSet->GetDeviceData(), 
                     nodes->GetPrimitiveInfoData() + activeIndex,
                     resultMin->GetDeviceData(), resultMax->GetDeviceData(),
                     activeRange);
                CHECK_FOR_CUDA_ERROR();

                /*                
                logger.info << nodes->ToString(activeIndex + 39) << logger.end;
                int primOffset = 626;
                int primRange = 6;

                logger.info << "primMin: " << Convert::ToString(resultMin->GetDeviceData() + primOffset, 1) << 
                    ", " << Convert::ToString(resultMin->GetDeviceData() + primOffset + 2, 1) << 
                    ", " << Convert::ToString(resultMin->GetDeviceData() + primOffset + 5, 1) << logger.end;
                logger.info << "primMax: " << Convert::ToString(resultMax->GetDeviceData() + primOffset, 1) << 
                    ", " << Convert::ToString(resultMax->GetDeviceData() + primOffset + 2, 1) << 
                    ", " << Convert::ToString(resultMax->GetDeviceData() + primOffset + 5, 1) << logger.end;

                logger.info << "X splittingSets: " << Convert::ToString(splitTriangleSet->GetDeviceData() + primOffset, primRange) << logger.end;
                logger.info << "Y splittingSets: " << Convert::ToString(splitTriangleSet->GetDeviceData() + primOffset + triangles, primRange) << logger.end;
                logger.info << "Z splittingSets: " << Convert::ToString(splitTriangleSet->GetDeviceData() + primOffset + 2 * triangles, primRange) << logger.end;
                */

#if CPU_VERIFY
                CheckLowerPreprocess(activeIndex, activeRange);
#endif
            }

            void TriangleMap::CheckLowerPreprocess(int activeIndex, int activeRange){

                int leafIDs[activeRange];
                cudaMemcpy(leafIDs, upperNodeLeafList->GetDeviceData(), activeRange * sizeof(int), cudaMemcpyDeviceToHost);
                CHECK_FOR_CUDA_ERROR();

                char info[activeRange];
                int2 leafPrimInfo[activeRange];
                int left[activeRange];
                for (int i = 0; i < activeRange; ++i){
                    cudaMemcpy(info + i, nodes->GetInfoData() + leafIDs[i], sizeof(char), cudaMemcpyDeviceToHost);
                    cudaMemcpy(leafPrimInfo + i, nodes->GetPrimitiveInfoData() + leafIDs[i], sizeof(int2), cudaMemcpyDeviceToHost);
                    cudaMemcpy(left + i, nodes->GetLeftData() + leafIDs[i], sizeof(int), cudaMemcpyDeviceToHost);
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
                        logger.info << nodes->ToString(leafIDs[i]) << logger.end;
                        logger.info << nodes->ToString(activeIndex + i) << logger.end;
                        throw Exception("info not equal to PROXY.");
                    }
                    if (left[i] != activeIndex + i)
                        throw Exception("leaf not pointing to correct lower node");
                }

                for (int i = 0; i < activeRange; ++i){
                    if (lowerPrimInfo[i].x != leafPrimInfo[i].x)
                        throw Exception("Leaf node " + Utils::Convert::ToString(leafIDs[i]) +
                                        "'s index " + Utils::Convert::ToString(leafPrimInfo[i].x) + 
                                        " does not match lower node " + Utils::Convert::ToString(activeIndex + i) +
                                        "'s " + Utils::Convert::ToString(lowerPrimInfo[i].x));
                    if (bitcount(lowerPrimInfo[i].y) > leafPrimInfo[i].y)
                        throw Exception("Leaf node " + Utils::Convert::ToString(leafIDs[i]) +
                                        "'s size of " + Utils::Convert::ToString(leafPrimInfo[i].y) + 
                                        " does not match lower node " + Utils::Convert::ToString(activeIndex + i) +
                                        "'s bitmap " + BitmapToString(lowerPrimInfo[i].y));
                }

                // @TODO check split set
                
            }
            
            void TriangleMap::ProcessLowerNodes(int activeIndex, int activeRange, 
                                                int &childrenCreated){
                logger.info << "=== Process " << activeRange << " Lower Nodes Starting at " << activeIndex << " ===" << logger.end;

                cudaMemcpyToSymbol(d_activeNodeIndex, &activeIndex, sizeof(int));
                cudaMemcpyToSymbol(d_activeNodeRange, &activeRange, sizeof(int));

                childAreas->Extend(activeRange);
                childSets->Extend(activeRange);
                splitSide->Extend(activeRange);

                //CUDADataBlock<1, float> debug = CUDADataBlock<1, float>(activeRange);

                unsigned int blocks, threads;
                Calc1DKernelDimensions(activeRange, blocks, threads, 96);
                //blocks = 13; threads = 96;
                unsigned int smemSize = threads * TriangleNode::MAX_LOWER_SIZE * sizeof(float);
                logger.info << "<<<" << blocks << ", " << threads << ", " << smemSize << ">>>" << logger.end;
                CalcSAH<<<blocks, threads, smemSize>>>(nodes->GetInfoData() + activeIndex,
                                                       nodes->GetSplitPositionData() + activeIndex,
                                                       nodes->GetPrimitiveInfoData() + activeIndex,
                                                       nodes->GetSurfaceAreaData() + activeIndex,
                                                       resultMin->GetDeviceData(),
                                                       resultMax->GetDeviceData(),
                                                       splitTriangleSet->GetDeviceData(),
                                                       childAreas->GetDeviceData(),
                                                       childSets->GetDeviceData(),
                                                       splitSide->GetDeviceData());
                CHECK_FOR_CUDA_ERROR();

                logger.info << nodes->ToString(activeIndex) << logger.end;

                int primOffset = 0;
                int primRange = 8;

                logger.info << "primMin: " << Convert::ToString(resultMin->GetDeviceData() + primOffset, primRange) << logger.end;
                logger.info << "primMax: " << Convert::ToString(resultMax->GetDeviceData() + primOffset, primRange) << logger.end;

                logger.info << "X splittingSets: " << Convert::ToString(splitTriangleSet->GetDeviceData() + primOffset, primRange) << logger.end;
                logger.info << "Y splittingSets: " << Convert::ToString(splitTriangleSet->GetDeviceData() + primOffset + triangles, primRange) << logger.end;
                logger.info << "Z splittingSets: " << Convert::ToString(splitTriangleSet->GetDeviceData() + primOffset + 2 * triangles, primRange) << logger.end;

                logger.info << nodes->ToString(activeIndex+40) << logger.end;

                childrenCreated = 0;
            }

        }
    }
}
