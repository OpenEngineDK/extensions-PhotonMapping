// Triangle map balanced creator interface for CUDA
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Utils/CUDA/TriangleMapBalancedFast.h>

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

            namespace TMBF{
#include <Utils/CUDA/Kernels/LowerTriangleMap.h>
            }
            using namespace TMBF;

            __global__ void SplitNodes(int *indices,
                                       char *info,
                                       float *splitPoss,
                                       int2 *primitiveInfo,
                                       int2* children,
                                       float4 *primMin, float4* primMax,
                                       int4* splitTriangleSet,
                                       int* newIndices,
                                       unsigned int* validIndices){
                const int id = blockDim.x * blockIdx.x + threadIdx.x;
        
                if (id < d_activeNodeRange){
                    const int parentID = indices[id];
                    const int2 primInfo = primitiveInfo[parentID];

                    float relation = 0.0f;
                    int largestSetSize = TriangleNode::MAX_LOWER_SIZE;
                    int leftSet, rightSet;
                    char axis;
                    int splitIndex;
                    
                    int triangles = primInfo.y;
                    while(triangles){
                        int i = __ffs(triangles) - 1;
                        
                        CalcRelationForSets(splitTriangleSet[primInfo.x + i], primInfo.y,
                                            KDNode::X, primInfo.x + i,
                                            relation, largestSetSize,
                                            leftSet, rightSet,
                                            axis, splitIndex);
                        
                        CalcRelationForSets(splitTriangleSet[d_triangles + primInfo.x + i], primInfo.y,
                                            KDNode::Y, primInfo.x + i,
                                            relation, largestSetSize,
                                            leftSet, rightSet,
                                            axis, splitIndex);
            
                        CalcRelationForSets(splitTriangleSet[2 * d_triangles + primInfo.x + i], primInfo.y,
                                            KDNode::Z, primInfo.x + i,
                                            relation, largestSetSize,
                                            leftSet, rightSet,
                                            axis, splitIndex);
                        
                        triangles -= 1<<i;
                    }
                    
                    bool split = minLeafTriangles < largestSetSize * relation;
                    const int leftID = d_activeNodeIndex + id;
                    const int rightID = leftID + d_activeNodeRange;

                    if (split){
                        // Dump stuff and move on
                        float3 splitPositions;
                        if (splitIndex & 1<<31){
                            // A high splitplane was used
                            splitPositions = make_float3(primMax[splitIndex ^ 1<<31]);
                        }else{
                            // A low splitplane was used
                            splitPositions = make_float3(primMin[splitIndex]);
                        }
                        splitPoss[parentID] = axis == KDNode::X ? splitPositions.x : (axis == KDNode::Y ? splitPositions.y : splitPositions.z);
                        primitiveInfo[leftID] = make_int2(primInfo.x, leftSet);
                        primitiveInfo[rightID] = make_int2(primInfo.x, rightSet);
                        children[parentID] = make_int2(leftID, rightID);
                    }
                    
                    newIndices[id] = leftID;
                    newIndices[d_activeNodeRange + id] = rightID;
                    info[parentID] = split ? axis : KDNode::LEAF;
                    validIndices[id] = validIndices[d_activeNodeRange + id] = split;
                }
            }

            TriangleMapBalancedFast::TriangleMapBalancedFast()
                : ITriangleMapCreator() {
                
                cutCreateTimer(&timerID);

                logger.info << "Create indexed balanced lower tree creator" << logger.end;

                splitTriangleSet =  new CUDADataBlock<1, int4>(1);
                newIndices =  new CUDADataBlock<1, int>(1);
                validIndices =  new CUDADataBlock<1, unsigned int>(1);
                
                compactConfig.algorithm = CUDPP_COMPACT;
                compactConfig.op = CUDPP_ADD;
                compactConfig.datatype = CUDPP_INT;
                compactConfig.options = CUDPP_OPTION_FORWARD;
                compactSize = 262144;
                CUDPPResult res = cudppPlan(&compactHandle, compactConfig, compactSize, 1, 0);
                if (CUDPP_SUCCESS != res)
                    throw Core::Exception("Error creating CUDPP compactPlan for Triangle Map balanced Creator");
                cudaMalloc(&numValid, sizeof(size_t));
            }

            TriangleMapBalancedFast::~TriangleMapBalancedFast(){
                if (splitTriangleSet) delete splitTriangleSet;
                if (newIndices) delete newIndices;
                if (validIndices) delete validIndices;
                cudaFree(numValid);
            }
            
            void TriangleMapBalancedFast::Create(TriangleMap* map, 
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

                CUDADataBlock<1, int> *tempIndices = newIndices;
                indices = upperLeafIDs;

                START_TIMER(timerID); 
                while (activeRange > 0){
                    ProcessLowerNodes(activeIndex, activeRange,
                                      map, childrenCreated);
                    
                    activeIndex = map->nodes->GetSize();
                    activeRange = childrenCreated;
                }
                PRINT_TIMER(timerID, "Process lower nodes into balanced subtrees");

                newIndices = tempIndices;
            }
            
            void TriangleMapBalancedFast::PreprocessLowerNodes(int activeIndex, int activeRange, 
                                                               TriangleMap* map, CUDADataBlock<1, int>* upperLeafIDs){
                int triangles = primMin->GetSize();
                logger.info << "=== Preprocess " << activeRange << " Lower Nodes Starting at " << activeIndex << " === with " << triangles << " primitives" << logger.end;
                
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
            
            void TriangleMapBalancedFast::ProcessLowerNodes(int activeIndex, int activeRange, 
                                                            TriangleMap* map, int &childrenCreated){

                logger.info << "=== Process " << activeRange << " Lower Nodes Starting at " << activeIndex << " ===" << logger.end;
                
                TriangleNode* nodes = map->nodes;

                cudaMemcpyToSymbol(d_activeNodeIndex, &activeIndex, sizeof(int));
                cudaMemcpyToSymbol(d_activeNodeRange, &activeRange, sizeof(int));

                nodes->Extend(activeIndex + 2 * activeRange);
                indices->Resize(activeRange * 2);
                newIndices->Resize(activeRange * 2, false);
                validIndices->Resize(activeRange * 2, false);

                int* index = indices->GetData();

                unsigned int blocks, threads;
                Calc1DKernelDimensions(activeRange, blocks, threads, 128);
                SplitNodes<<<blocks, threads>>>(indices->GetDeviceData(),
                                                nodes->GetInfoData(),
                                                nodes->GetSplitPositionData(),
                                                nodes->GetPrimitiveInfoData(),
                                                nodes->GetChildrenData(),
                                                primMin->GetDeviceData(),
                                                primMax->GetDeviceData(),
                                                splitTriangleSet->GetDeviceData(),
                                                indices->GetDeviceData(),
                                                validIndices->GetDeviceData());
                CHECK_FOR_CUDA_ERROR();

                //logger.info << "indices: " << Convert::ToString(indices->GetDeviceData(), 30) << logger.end;
                //logger.info << "validIndices: " << Convert::ToString((int*)validIndices->GetDeviceData(), 30) << logger.end;

                // Compact indices
                cudppCompact(compactHandle, newIndices->GetDeviceData(), numValid, 
                             indices->GetDeviceData(), validIndices->GetDeviceData(), activeRange * 2);
                CHECK_FOR_CUDA_ERROR();

                //logger.info << "newIndices: " << Convert::ToString(newIndices->GetDeviceData(), 30) << logger.end;

                std::swap(indices, newIndices);

                size_t hat;
                cudaMemcpy(&hat, numValid, sizeof(size_t), cudaMemcpyDeviceToHost);
                childrenCreated = hat;

                //logger.info << childrenCreated << logger.end;

                /*
                for (int i = 0; i < 20; ++i)
                    logger.info << nodes->ToString(index[i]) << logger.end;
                */
                //exit(0);
            }
            
        }
    }
}
