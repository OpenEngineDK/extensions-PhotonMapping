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
#include <Utils/CUDA/Convert.h>
#include <Utils/CUDA/Utils.h>
#include <Utils/CUDA/IntersectionTests.h>

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
                : ITriangleMapCreator(), traversalCost(24.0f) {

                cutCreateTimer(&timerID);

                splitTriangleSet =  new CUDADataBlock<1, KDNode::bitmap4>(1);
                primAreas = new CUDADataBlock<1, float>(1);
                childAreas = new CUDADataBlock<1, float2>(1);
                childSets = new CUDADataBlock<1, KDNode::bitmap2>(1);
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
            
            TriangleMapSAHCreator::~TriangleMapSAHCreator() {
                if (splitTriangleSet) delete splitTriangleSet;
                if (primAreas) delete primAreas;
                if (childAreas) delete childAreas;
                if (childSets) delete childSets;
                if (splitSide) delete splitSide;
                if (splitAddr) delete splitAddr;
            }

            void TriangleMapSAHCreator::Create(TriangleMap* map,
                                               CUDADataBlock<1, int>* upperLeafIDs){
                
                primMin = map->primMin;
                primMax = map->primMax;

                SetPropagateBoundingBox(map->GetPropagateBoundingBox());

                int activeIndex = map->nodes->GetSize(); int activeRange = upperLeafIDs->GetSize();
                int childrenCreated;

                int triangles = map->primMin->GetSize();
                cudaMemcpyToSymbol(d_triangles, &triangles, sizeof(int));

                //START_TIMER(timerID); 
                PreprocessLowerNodes(activeIndex, activeRange, map, upperLeafIDs);
                //PRINT_TIMER(timerID, "Preprocess lower nodes using SAH");

                //START_TIMER(timerID); 
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
                //PRINT_TIMER(timerID, "Process lower nodes using SAH");
            }

            void TriangleMapSAHCreator::PreprocessLowerNodes(int activeIndex, int activeRange, 
                                      TriangleMap* map, CUDADataBlock<1, int>* upperLeafIDs) {
                int triangles = primMin->GetSize();
                //logger.info << "=== Preprocess " << activeRange << " Lower Nodes Starting at " << activeIndex << " === with " << triangles << " indices" << logger.end;
                
                GeometryList* geom = map->GetGeometry();

                primAreas->Extend(triangles);

                unsigned int blocks, threads, smemSize;
                Calc1DKernelDimensions(triangles, blocks, threads);
                CalcSurfaceArea<<<blocks, threads>>>(map->GetPrimitiveIndices()->GetDeviceData(),
                                                     geom->GetP0Data(),
                                                     geom->GetP1Data(),
                                                     geom->GetP2Data(),
                                                     primAreas->GetDeviceData(),
                                                     triangles);
                CHECK_FOR_CUDA_ERROR();

                TriangleNode* nodes = map->nodes;

                splitTriangleSet->Extend(triangles * 3);
                
                Calc1DKernelDimensions(activeRange, blocks, threads);
                
                PreprocesLowerNodes<<<blocks, threads>>>(upperLeafIDs->GetDeviceData(),
                                                         nodes->GetPrimitiveIndexData(),
                                                         nodes->GetPrimitiveBitmapData(),
                                                         nodes->GetSurfaceAreaData(),
                                                         primAreas->GetDeviceData(),
                                                         activeRange);
                CHECK_FOR_CUDA_ERROR();

                unsigned int smemPrThread = sizeof(float3) + sizeof(float3);
                Calc1DKernelDimensionsWithSmem(activeRange * TriangleNode::MAX_LOWER_SIZE, smemPrThread, 
                                               blocks, threads, smemSize, 128);
                CreateSplittingPlanes<<<blocks, threads, smemSize>>>
                    (upperLeafIDs->GetDeviceData(),
                     nodes->GetPrimitiveIndexData(),
                     nodes->GetPrimitiveBitmapData(),
                     primMin->GetDeviceData(), primMax->GetDeviceData(),
                     splitTriangleSet->GetDeviceData(), 
                     activeRange);
                CHECK_FOR_CUDA_ERROR();

#if CPU_VERIFY
                CheckPreprocess(activeIndex, activeRange, map, upperLeafIDs);
#endif

            }
                
            void TriangleMapSAHCreator::ProcessLowerNodes(int activeIndex, int activeRange, 
                                                          TriangleMap* map, CUDADataBlock<1, int>* upperLeafIDs, 
                                                          int &childrenCreated) {
                /*
                if (upperLeafIDs)
                    logger.info << "=== Process " << activeRange << " Lower Nodes from Indices ===" << logger.end;
                else
                    logger.info << "=== Process " << activeRange << " Lower Nodes Starting at " << activeIndex << " ===" << logger.end;
                */

                TriangleNode* nodes = map->nodes;
                
                cudaMemcpyToSymbol(d_activeNodeIndex, &activeIndex, sizeof(int));
                cudaMemcpyToSymbol(d_activeNodeRange, &activeRange, sizeof(int));

                childAreas->Extend(activeRange);
                childSets->Extend(activeRange);
                splitSide->Extend(activeRange+1);
                splitAddr->Extend(activeRange+1);

                unsigned int blocks, threads, smemSize;
                unsigned int smemPrThread = TriangleNode::MAX_LOWER_SIZE * sizeof(float);
                Calc1DKernelDimensionsWithSmem(activeRange, smemPrThread, 
                                               blocks, threads, smemSize, 96);
                //logger.info << "<<<" << blocks << ", " << threads << ", " << smemSize << ">>>" << logger.end;
                if (upperLeafIDs)
                    CalcSAH<true><<<blocks, threads, smemSize>>>(upperLeafIDs->GetDeviceData(), 
                                                                 nodes->GetInfoData(),
                                                                 nodes->GetSplitPositionData(),
                                                                 nodes->GetPrimitiveIndexData(),
                                                                 nodes->GetPrimitiveBitmapData(),
                                                                 nodes->GetSurfaceAreaData(),
                                                                 primAreas->GetDeviceData(),
                                                                 primMin->GetDeviceData(),
                                                                 primMax->GetDeviceData(),
                                                                 splitTriangleSet->GetDeviceData(),
                                                                 childAreas->GetDeviceData(),
                                                                 childSets->GetDeviceData(),
                                                                 splitSide->GetDeviceData(),
                                                                 traversalCost);
                else
                    CalcSAH<false><<<blocks, threads, smemSize>>>(NULL, 
                                                                  nodes->GetInfoData(),
                                                                  nodes->GetSplitPositionData(),
                                                                  nodes->GetPrimitiveIndexData(),
                                                                  nodes->GetPrimitiveBitmapData(),
                                                                  nodes->GetSurfaceAreaData(),
                                                                  primAreas->GetDeviceData(),
                                                                  primMin->GetDeviceData(),
                                                                  primMax->GetDeviceData(),
                                                                  splitTriangleSet->GetDeviceData(),
                                                                  childAreas->GetDeviceData(),
                                                                  childSets->GetDeviceData(),
                                                                  splitSide->GetDeviceData(),
                                                                  traversalCost);
                CHECK_FOR_CUDA_ERROR();

                cudppScan(scanHandle, splitAddr->GetDeviceData(), splitSide->GetDeviceData(), activeRange+1);
                CHECK_FOR_CUDA_ERROR();

                int splits;
                cudaMemcpy(&splits, splitAddr->GetDeviceData() + activeRange, sizeof(int), cudaMemcpyDeviceToHost);
                nodes->Extend(nodes->GetSize() + 2 * splits);

                Calc1DKernelDimensions(activeRange, blocks, threads);
                if (upperLeafIDs)
                    CreateLowerSAHChildren<true><<<blocks, threads>>>(upperLeafIDs->GetDeviceData(), 
                                                                      splitSide->GetDeviceData(),
                                                                      splitAddr->GetDeviceData(),
                                                                      childAreas->GetDeviceData(),
                                                                      childSets->GetDeviceData(),
                                                                      nodes->GetSurfaceAreaData(),
                                                                      nodes->GetPrimitiveIndexData(),
                                                                      nodes->GetPrimitiveAmountData(),
                                                                      nodes->GetChildrenData(),
                                                                      splits);
                else
                    CreateLowerSAHChildren<false><<<blocks, threads>>>(NULL, splitSide->GetDeviceData(),
                                                                       splitAddr->GetDeviceData(),
                                                                       childAreas->GetDeviceData(),
                                                                       childSets->GetDeviceData(),
                                                                       nodes->GetSurfaceAreaData(),
                                                                       nodes->GetPrimitiveIndexData(),
                                                                       nodes->GetPrimitiveAmountData(),
                                                                       nodes->GetChildrenData(),
                                                                       splits);
                CHECK_FOR_CUDA_ERROR();

                childrenCreated = splits * 2;
                
                if (propagateAabbs && childrenCreated > 0){
                    
                    // @TODO propagate downwards or upwards? Test
                    // which is fastest (for non trivial splits
                    // sherlock
                    if (upperLeafIDs){
                        PropagateAabbToChildren<true><<<blocks, threads>>>(upperLeafIDs->GetDeviceData(), 
                                                                           nodes->GetInfoData(), nodes->GetSplitPositionData(),
                                                                           nodes->GetAabbMinData(), nodes->GetAabbMaxData(), 
                                                                           nodes->GetChildrenData());
                    }else
                        PropagateAabbToChildren<false><<<blocks, threads>>>(NULL, nodes->GetInfoData(), nodes->GetSplitPositionData(),
                                                                            nodes->GetAabbMinData(), nodes->GetAabbMaxData(), 
                                                                            nodes->GetChildrenData());
                    CHECK_FOR_CUDA_ERROR();
                }
            }

            void TriangleMapSAHCreator::CheckPreprocess(int activeIndex, int activeRange, 
                                 TriangleMap* map, Resources::CUDA::CUDADataBlock<1, int>* leafIDs) {

                throw Exception("CheckPreprocess was broken by removing PROXY");

                /*
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
                */
                // @TODO check split set

            }
            
        }
    }
}
