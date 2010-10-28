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
                logger.info << "=== Preprocess " << activeRange << " Lower Nodes Starting at " << activeIndex << " ===" << logger.end;
                
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

                Calc1DKernelDimensions(triangles, blocks, threads, 256);
                unsigned int smemSize = threads * (sizeof(float3) + sizeof(float4));
                CreateSplittingPlanes<<<blocks, threads, smemSize>>>
                    (splitTriangleSet->GetDeviceData(), 
                     nodes->GetPrimitiveInfoData(),
                     resultMin->GetDeviceData(), resultMax->GetDeviceData(),
                     activeRange);
                CHECK_FOR_CUDA_ERROR();

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

                for (int i = 0; i < activeRange; ++i){
                    if (info[i] != KDNode::PROXY)
                        throw Exception("info not equal to PROXY.");
                    if (left[i] != activeIndex + i)
                        throw Exception("leaf not pointing to correct lower node");
                }

                int2 lowerPrimInfo[activeRange];
                cudaMemcpy(lowerPrimInfo, nodes->GetPrimitiveInfoData() + activeIndex, activeRange * sizeof(int2), cudaMemcpyDeviceToHost);
                CHECK_FOR_CUDA_ERROR();

                for (int i = 0; i < activeRange; ++i){
                    if (lowerPrimInfo[i].x != leafPrimInfo[i].x)
                        throw Exception("Leaf node " + Utils::Convert::ToString(leafIDs[i]) +
                                        "'s index " + Utils::Convert::ToString(leafPrimInfo[i].x) + 
                                        " does not match lower node " + Utils::Convert::ToString(activeIndex + i) +
                                        "'s " + Utils::Convert::ToString(lowerPrimInfo[i].x));
                    if (bitcount(lowerPrimInfo[i].y) != leafPrimInfo[i].y)
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

                childrenCreated = 0;
            }

        }
    }
}
