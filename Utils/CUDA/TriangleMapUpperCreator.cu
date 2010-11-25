// Triangle map upper node creator interface for CUDA
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Utils/CUDA/TriangleMapUpperCreator.h>

#include <Scene/TriangleNode.h>
#include <Utils/CUDA/Convert.h>
#include <Utils/CUDA/GeometryList.h>
#include <Utils/CUDA/TriangleMap.h>
#include <Utils/CUDA/Utils.h>

#include <Utils/CUDA/Kernels/PhotonMapDeviceVars.h>

namespace OpenEngine {    
    using namespace Scene;
    namespace Utils {
        namespace CUDA {

            using namespace Kernels;

            TriangleMapUpperCreator::TriangleMapUpperCreator()
                : ITriangleMapCreator(), emptySpaceThreshold(0.5f) {
                
                cutCreateTimer(&timerID);

                aabbMin = new CUDADataBlock<1, float4>(1);
                aabbMax = new CUDADataBlock<1, float4>(1);
                tempAabbMin = new CUDADataBlock<1, float4>(1);
                tempAabbMax = new CUDADataBlock<1, float4>(1);

                segments = Segments(1);
                nodeSegments = new CUDADataBlock<1, int>(1);

                splitSide = new CUDADataBlock<1, int>(1);
                splitAddr = new CUDADataBlock<1, int>(1);
                leafSide = new CUDADataBlock<1, int>(1);
                leafAddr = new CUDADataBlock<1, int>(1);
                emptySpacePlanes = new CUDADataBlock<1, char>(1);
                emptySpaceNodes = new CUDADataBlock<1, int>(1);
                emptySpaceAddrs = new CUDADataBlock<1, int>(1);
                nodeIndices = new CUDADataBlock<1, int>(1);
                childSize = new CUDADataBlock<1, int2>(1);

                // CUDPP doesn't handle removing handles well, so we
                // define them to accept some arbitrary high number of
                // elements here.
                scanConfig.algorithm = CUDPP_SCAN;
                scanConfig.op = CUDPP_ADD;
                scanConfig.datatype = CUDPP_INT;
                scanConfig.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
                scanSize = 262144;
                
                CUDPPResult res = cudppPlan(&scanHandle, scanConfig, scanSize, 1, 0);
                if (CUDPP_SUCCESS != res)
                    throw Core::Exception("Error creating CUDPP scanPlan for Triangle Map");

                scanInclConfig.algorithm = CUDPP_SCAN;
                scanInclConfig.op = CUDPP_ADD;
                scanInclConfig.datatype = CUDPP_INT;
                scanInclConfig.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
                scanInclSize = 262144;

                res = cudppPlan(&scanInclHandle, scanInclConfig, scanInclSize, 1, 0);
                if (CUDPP_SUCCESS != res)
                    throw Core::Exception("Error creating CUDPP inclusive scanPlan for Triangle Map");
            }

            TriangleMapUpperCreator::~TriangleMapUpperCreator(){
                
                if (aabbMin) delete aabbMin;
                if (aabbMax) delete aabbMax;
                if (tempAabbMin) delete tempAabbMin;
                if (tempAabbMax) delete tempAabbMax;

                if (nodeSegments) delete nodeSegments;

                if (splitSide) delete splitSide;
                if (splitAddr) delete splitAddr;
                if (leafSide) delete leafSide;
                if (leafAddr) delete leafAddr;
                if (emptySpacePlanes) delete emptySpacePlanes;
                if (emptySpaceNodes) delete emptySpaceNodes;
                if (emptySpaceAddrs) delete emptySpaceAddrs;
                if (nodeIndices) delete nodeIndices;
                if (childSize) delete childSize;
            }

            namespace KernelsHat {
                #include <Utils/CUDA/Kernels/TriangleUpper.h>
                #include <Utils/CUDA/Kernels/TriangleUpperSegment.h>
                #include <Utils/CUDA/Kernels/ReduceSegments.h>
                #include <Utils/CUDA/Kernels/TriangleUpperChildren.h>
                #include <Utils/CUDA/Kernels/TriangleKernels.h>
                #include <Utils/CUDA/Kernels/EmptySpaceSplitting.h>
            }
            using namespace KernelsHat;

            void TriangleMapUpperCreator::Create(TriangleMap* map, 
                                                 CUDADataBlock<1, int>* upperLeafIDs){

                this->map = map;

                primMin = map->primMin;
                primMax = map->primMax;
                primIndices = map->primIndices;
                leafIDs = map->leafIDs;
                
                int activeIndex = 0, activeRange = 1;
                int childrenCreated;
                int triangles = map->GetGeometry()->GetSize();

                cudaMemcpyToSymbol(d_emptySpaceThreshold, &emptySpaceThreshold, sizeof(float));

                primMin->Extend(0);
                primMax->Extend(0);
                leafIDs->Extend(0);

                // Setup root node!
                int2 i = make_int2(0, triangles);
                cudaMemcpy(map->nodes->GetPrimitiveInfoData(), &i, sizeof(int2), cudaMemcpyHostToDevice);
                int parent = 0;
                cudaMemcpy(map->nodes->GetParentData(), &parent, sizeof(int), cudaMemcpyHostToDevice);
                float4 zero = make_float4(0.0f);
                cudaMemcpy(map->nodes->GetAabbMinData(), &zero, sizeof(float4), cudaMemcpyHostToDevice);
                cudaMemcpy(map->nodes->GetAabbMaxData(), &zero, sizeof(float4), cudaMemcpyHostToDevice);
                map->nodes->Resize(1);

                // Setup bounding box info
                aabbMin->Extend(triangles);
                aabbMax->Extend(triangles);
                
                unsigned int blocks, threads;
                Calc1DKernelDimensions(triangles, blocks, threads);
                CalcPrimitiveAabb<<<blocks, threads>>>(map->GetGeometry()->GetP0Data(),
                                                       map->GetGeometry()->GetP1Data(),
                                                       map->GetGeometry()->GetP2Data(),
                                                       aabbMin->GetDeviceData(),
                                                       aabbMax->GetDeviceData(),
                                                       triangles);
                CHECK_FOR_CUDA_ERROR();                

                START_TIMER(timerID);
                while (activeRange > 0){
                    ProcessNodes(activeIndex, activeRange, 
                                 childrenCreated);

                    activeIndex = map->nodes->GetSize() - childrenCreated;
                    activeRange = childrenCreated;
                }
                PRINT_TIMER(timerID, "triangle upper map");

                // Extract indices from primMin.
                primIndices->Extend(primMin->GetSize(), false);
                Calc1DKernelDimensions(primMin->GetSize(), blocks, threads, 128);
                ExtractIndexFromAabb<<<blocks, threads>>>(primMin->GetDeviceData(), 
                                                          primIndices->GetDeviceData(),
                                                          primMin->GetSize());
                CHECK_FOR_CUDA_ERROR();
            }
            
            void TriangleMapUpperCreator::ProcessNodes(int activeIndex, int activeRange, 
                                                       int &childrenCreated){
                int triangles = aabbMin->GetSize();
                logger.info << "=== Process " << activeRange << " Upper Nodes Starting at " << activeIndex << " === with " << triangles << " primitives" << logger.end;

                // Copy bookkeeping to symbols
                cudaMemcpyToSymbol(d_activeNodeIndex, &activeIndex, sizeof(int));
                cudaMemcpyToSymbol(d_activeNodeRange, &activeRange, sizeof(int));
                cudaMemcpyToSymbol(d_triangles, &triangles, sizeof(int));
                CHECK_FOR_CUDA_ERROR();

                Segment(activeIndex, activeRange);
                CHECK_FOR_CUDA_ERROR();

                // Calculate aabb
                ReduceAabb(activeIndex, activeRange);
                CHECK_FOR_CUDA_ERROR();

                // Calculate children placement
                CreateChildren(activeIndex, activeRange, childrenCreated);
            }

            void TriangleMapUpperCreator::Segment(int activeIndex, int activeRange){
                nodeSegments->Extend(activeRange+1);

                unsigned int blocks, threads;
                Calc1DKernelDimensions(activeRange, blocks, threads);
                NodeSegments<<<blocks, threads>>>(map->nodes->GetPrimitiveInfoData() + activeIndex,
                                                  nodeSegments->GetDeviceData());

                CHECK_FOR_CUDA_ERROR();
                cudppScan(scanHandle, nodeSegments->GetDeviceData(), nodeSegments->GetDeviceData(), activeRange+1);

                int amountOfSegments;
                cudaMemcpy(&amountOfSegments, nodeSegments->GetDeviceData() + activeRange, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpyToSymbol(d_segments, nodeSegments->GetDeviceData() + activeRange, sizeof(int), 0, cudaMemcpyDeviceToDevice);
                CHECK_FOR_CUDA_ERROR();

                segments.Extend(amountOfSegments);

                cudaMemset(segments.GetOwnerData(), 0, amountOfSegments * sizeof(int));
                MarkOwnerStart<<<blocks, threads>>>(segments.GetOwnerData(),
                                                    nodeSegments->GetDeviceData());
                CHECK_FOR_CUDA_ERROR();

                cudppScan(scanInclHandle, segments.GetOwnerData(), segments.GetOwnerData(), amountOfSegments);

                Calc1DKernelDimensions(amountOfSegments, blocks, threads);
                CalcSegmentPrimitives<<<blocks, threads>>>(segments.GetOwnerData(),
                                                           nodeSegments->GetDeviceData(),
                                                           map->nodes->GetPrimitiveInfoData(),
                                                           segments.GetPrimitiveInfoData());
                CHECK_FOR_CUDA_ERROR();
            }
                
            void TriangleMapUpperCreator::ReduceAabb(int &activeIndex, int activeRange){
                // Reduce aabb pr segment
                unsigned int blocks = segments.size;
                unsigned int threads = Segments::SEGMENT_SIZE;
                unsigned int smemSize = 2 * sizeof(float4) * segments.SEGMENT_SIZE;

                ReduceSegmentsShared<<<blocks, threads, smemSize>>>(segments.GetPrimitiveInfoData(),
                                                                   aabbMin->GetDeviceData(), aabbMax->GetDeviceData(),
                                                                   segments.GetAabbMinData(), segments.GetAabbMaxData());
                CHECK_FOR_CUDA_ERROR();

#if CPU_VERIFY
                float4 *finalMin, *finalMax;
                CheckSegmentReduction(activeIndex, activeRange,
                                      segments, &finalMin, &finalMax);
#endif

                tempAabbMin->Resize(activeRange, false);
                tempAabbMax->Resize(activeRange, false);
                
                Calc1DKernelDimensions(activeRange, blocks, threads);
                AabbMemset<<<blocks, threads>>>(tempAabbMin->GetDeviceData(),
                                                tempAabbMax->GetDeviceData());
                CHECK_FOR_CUDA_ERROR();
                
                Calc1DKernelDimensions(segments.GetSize(), blocks, threads);
                for (int i = 0; i < blocks; ++i){
                    int segs = segments.GetSize() - i * threads;
                    cudaMemcpyToSymbol(d_segments, &segs, sizeof(int));
                    FinalSegmentedReduce<<<1, threads>>>(segments.GetAabbMinData() + i * threads,
                                                         segments.GetAabbMaxData() + i * threads,
                                                         segments.GetOwnerData() + i * threads,
                                                         tempAabbMin->GetDeviceData(),
                                                         tempAabbMax->GetDeviceData());
                }
                int segs = segments.GetSize();
                cudaMemcpyToSymbol(d_segments, &segs, sizeof(int));
                CHECK_FOR_CUDA_ERROR();

                // Calculate empty space splitting planes before copying aabbs to nodes.
                CreateEmptySplits(activeIndex, activeRange);

                cudaMemcpy(map->nodes->GetAabbMinData() + activeIndex, 
                           tempAabbMin->GetDeviceData(), 
                           activeRange * sizeof(float4), cudaMemcpyDeviceToDevice);
                cudaMemcpy(map->nodes->GetAabbMaxData() + activeIndex, 
                           tempAabbMax->GetDeviceData(), 
                           activeRange * sizeof(float4), cudaMemcpyDeviceToDevice);


#if CPU_VERIFY
                CheckFinalReduction(activeIndex, activeRange, map->nodes, 
                                    finalMin, finalMax);
#endif

                // Calc splitting planes.
                Calc1DKernelDimensions(activeRange, blocks, threads);
                CalcUpperNodeSplitInfo<<<blocks, threads>>>(map->nodes->GetAabbMinData() + activeIndex,
                                                            map->nodes->GetAabbMaxData() + activeIndex,
                                                            map->nodes->GetSplitPositionData() + activeIndex,
                                                            map->nodes->GetInfoData() + activeIndex);
                CHECK_FOR_CUDA_ERROR();
            }

            void TriangleMapUpperCreator::CreateEmptySplits(int &activeIndex, int activeRange){
                bool createdEmptySplits = false;
                cudaMemcpyToSymbol(d_createdEmptySplits, &createdEmptySplits, sizeof(bool));
                emptySpacePlanes->Resize(activeRange, false);
                emptySpaceNodes->Resize(activeRange+1, false);
                
                unsigned int blocks, threads;
                Calc1DKernelDimensions(activeRange, blocks, threads);
                CalcEmptySpaceSplits<<<blocks, threads>>>(map->nodes->GetAabbMinData() + activeIndex,
                                                          map->nodes->GetAabbMaxData() + activeIndex,
                                                          tempAabbMin->GetDeviceData(), 
                                                          tempAabbMax->GetDeviceData(), 
                                                          emptySpacePlanes->GetDeviceData(),
                                                          emptySpaceNodes->GetDeviceData());
                CHECK_FOR_CUDA_ERROR();

                cudaMemcpyFromSymbol(&createdEmptySplits, d_createdEmptySplits, sizeof(bool));

                if (createdEmptySplits){
                    emptySpaceAddrs->Resize(activeRange+1, false);
                    cudppScan(scanHandle, emptySpaceAddrs->GetDeviceData(), emptySpaceNodes->GetDeviceData(), activeRange+1);

                    int emptyNodes;
                    cudaMemcpy(&emptyNodes, emptySpaceAddrs->GetDeviceData() + emptySpaceAddrs->GetSize()-1, sizeof(int), cudaMemcpyDeviceToHost);
                    CHECK_FOR_CUDA_ERROR();
                    logger.info << "Empty space " << emptyNodes << logger.end;

                    // Move nodes to make room for empty space nodes.
                    // That means moving primitiveInfo, using childSize as temp storage
                    // And moving parents, using splitSide as temp storage

                    //logger.info << "primInfo: " << Convert::ToString(map->nodes->GetPrimitiveInfoData() + activeIndex, activeRange) << logger.end;

                    childSize->Resize(activeRange);
                    splitSide->Resize(activeRange);
                    
                    cudaMemcpy(childSize->GetDeviceData(), map->nodes->GetPrimitiveInfoData() + activeIndex, activeRange * sizeof(int2), cudaMemcpyDeviceToDevice);
                    cudaMemcpy(splitSide->GetDeviceData(), map->nodes->GetParentData() + activeIndex, activeRange * sizeof(int), cudaMemcpyDeviceToDevice);

                    activeIndex += emptyNodes;
                    cudaMemcpyToSymbol(d_activeNodeIndex, &activeIndex, sizeof(int));
                    map->nodes->Resize(map->nodes->GetSize() + emptyNodes);

                    cudaMemcpy(map->nodes->GetPrimitiveInfoData() + activeIndex, childSize->GetDeviceData(), activeRange * sizeof(int2), cudaMemcpyDeviceToDevice);
                    cudaMemcpy(map->nodes->GetParentData() + activeIndex, splitSide->GetDeviceData(), activeRange * sizeof(int), cudaMemcpyDeviceToDevice);
                    segments.IncreaseNodeIDs(emptyNodes);
                    
                    //logger.info << "primInfo: " << Convert::ToString(map->nodes->GetPrimitiveInfoData() + activeIndex, activeRange) << logger.end;

                    // Create the empty space nodes
                    //logger.info << "parents: " << Convert::ToString(map->nodes->GetParentData() + activeIndex, activeRange) << logger.end;
                    int parents[activeRange];
                    cudaMemcpy(parents, map->nodes->GetParentData() + activeIndex, activeRange * sizeof(int), cudaMemcpyDeviceToHost);
                    int minNode = activeRange + activeIndex, maxNode = 0;
                    for (int i = 0; i < activeRange; ++i){
                        minNode = min(minNode, parents[i]);
                        maxNode = max(maxNode, parents[i]);
                    }
                        
                    //logger.info << "children from " << minNode << ": " << Convert::ToString(map->nodes->GetChildrenData() + minNode, maxNode-minNode) << logger.end;
                    
                    CorrectParentPointer<<<blocks, threads>>>(map->nodes->GetParentData(), 
                                                              map->nodes->GetChildrenData(),
                                                              emptyNodes);
                    CHECK_FOR_CUDA_ERROR();

                    //logger.info << "parents: " << Convert::ToString(map->nodes->GetParentData() + activeIndex, activeRange) << logger.end;
                    //logger.info << "children from " << minNode << ": " << Convert::ToString(map->nodes->GetChildrenData() + minNode, maxNode-minNode) << logger.end;

                    CheckEmptySpaceSplitting(activeIndex, activeRange);

                    //exit(0);
                }
            }

            void TriangleMapUpperCreator::CheckEmptySpaceSplitting(int activeIndex, int activeRange){
                
                int parents[activeRange];
                cudaMemcpy(parents, map->nodes->GetParentData() + activeIndex, activeRange * sizeof(int), cudaMemcpyDeviceToHost);
                int minNode = activeRange + activeIndex, maxNode = 0;
                for (int i = 0; i < activeRange; ++i){
                    minNode = min(minNode, parents[i]);
                    maxNode = max(maxNode, parents[i]);
                }
                
                int childrenRange = maxNode - minNode + 1;
                int2 children[childrenRange];
                cudaMemcpy(children, map->nodes->GetChildrenData() + minNode, childrenRange * sizeof(int2), cudaMemcpyDeviceToHost);
                
                /*
                logger.info << minNode << ", maxNode " << maxNode << logger.end;

                logger.info << "Parents: " << parents[0];
                for (int i = 1; i < activeRange; i++){
                    logger.info << ", " << parents[i];
                }
                logger.info << logger.end;

                logger.info << "Children: " << Convert::ToString(children[0]);
                for (int i = 1; i < childrenRange; i++){
                    logger.info << ", " << Convert::ToString(children[i]);
                }
                logger.info << logger.end;
                */

                for (int i = 0; i < activeRange; i++){
                    int parent = parents[i] - minNode;
                    int2 childIDs = children[parent];
                    
                    if (!(childIDs.x == i + activeIndex || childIDs.y == i + activeIndex)){
                        logger.info << "parent: " << parent << ", childIDs: " << Convert::ToString(childIDs) << ", activeIndex " << activeIndex << logger.end;
                        throw Exception("Not moved correctly");
                    }
                }

            }

            void TriangleMapUpperCreator::CheckSegmentReduction(int activeIndex, int activeRange,
                                                                Segments &segments, 
                                                                float4 **finalMin, 
                                                                float4 **finalMax){
                int2 info[segments.GetSize()];
                cudaMemcpy(info, segments.GetPrimitiveInfoData(), 
                           segments.GetSize() * sizeof(int2), cudaMemcpyDeviceToHost);

                float4 segMin[segments.GetSize()];
                cudaMemcpy(segMin, segments.GetAabbMinData(), 
                           segments.GetSize() * sizeof(float4), cudaMemcpyDeviceToHost);
                float4 segMax[segments.GetSize()];
                cudaMemcpy(segMax, segments.GetAabbMaxData(), 
                           segments.GetSize() * sizeof(float4), cudaMemcpyDeviceToHost);

                for (int i = 0; i < segments.GetSize(); ++i){
                    int index = info[i].x;
                    int range = info[i].y;

                    float4 cpuMin[range];
                    cudaMemcpy(cpuMin, aabbMin->GetDeviceData() + index, 
                               range * sizeof(float4), cudaMemcpyDeviceToHost);
                    float4 cpuMax[range];
                    cudaMemcpy(cpuMax, aabbMax->GetDeviceData() + index, 
                               range * sizeof(float4), cudaMemcpyDeviceToHost);

                    for (int j = 1; j < range; ++j){
                        cpuMin[0] = min(cpuMin[0], cpuMin[j]);
                        cpuMax[0] = max(cpuMax[0], cpuMax[j]);
                    }
                    
                    if (cpuMin[0].x != segMin[i].x || cpuMin[0].y != segMin[i].y || cpuMin[0].z != segMin[i].z)
                        throw Core::Exception("aabbMin error at segment " + Utils::Convert::ToString(i) +
                                              ": CPU min " + Utils::CUDA::Convert::ToString(cpuMin[0])
                                              + ", GPU min " + Utils::CUDA::Convert::ToString(segMin[i]));

                    if (cpuMax[0].x != segMax[i].x || cpuMax[0].y != segMax[i].y || cpuMax[0].z != segMax[i].z)
                        throw Core::Exception("aabbMax error at segment " + Utils::Convert::ToString(i) +
                                              ": CPU max " + Utils::CUDA::Convert::ToString(cpuMax[0])
                                              + ", GPU max " + Utils::CUDA::Convert::ToString(segMax[i]));
                }

                int segOwner[segments.GetSize()];
                cudaMemcpy(segOwner, segments.GetOwnerData(), 
                           segments.GetSize() * sizeof(int), cudaMemcpyDeviceToHost);

                (*finalMin) = new float4[activeRange];
                (*finalMax) = new float4[activeRange];

                int owner0 = segOwner[0];
                float4 localMin = segMin[0];
                float4 localMax = segMax[0];
                for (int i = 1; i < segments.GetSize(); ++i){
                    int owner1 = segOwner[i];
                    if (owner0 != owner1){
                        (*finalMin)[owner0 - activeIndex] = localMin;
                         (*finalMax)[owner0 - activeIndex] = localMax;
                        owner0 = segOwner[i];
                        localMin = segMin[i];
                        localMax = segMax[i];
                    }else{
                        localMin = min(localMin, segMin[i]);
                        localMax = max(localMax, segMax[i]);
                    }
                }
                (*finalMin)[owner0 - activeIndex] = localMin;
                (*finalMax)[owner0 - activeIndex] = localMax;
            }

            void TriangleMapUpperCreator::CheckFinalReduction(int activeIndex, int activeRange,
                                                              TriangleNode* nodes, 
                                                              float4 *finalMin, 
                                                              float4 *finalMax){
                
                float4 gpuMin[activeRange];
                cudaMemcpy(gpuMin, nodes->GetAabbMinData() + activeIndex,
                           activeRange * sizeof(float4), cudaMemcpyDeviceToHost);
                float4 gpuMax[activeRange];
                cudaMemcpy(gpuMax, nodes->GetAabbMaxData() + activeIndex,
                           activeRange * sizeof(float4), cudaMemcpyDeviceToHost);

                for (int i = 0; i < activeRange; ++i){
                    if (finalMin[i].x != gpuMin[i].x || finalMin[i].y != gpuMin[i].y || finalMin[i].z != gpuMin[i].z)
                        throw Core::Exception("aabbMin error at node " + Utils::Convert::ToString(i + activeIndex) +
                                              ": CPU min " + Utils::CUDA::Convert::ToString(finalMin[i])
                                              + ", GPU min " + Utils::CUDA::Convert::ToString(gpuMin[i]));

                    if (finalMax[i].x != gpuMax[i].x || finalMax[i].y != gpuMax[i].y || finalMax[i].z != gpuMax[i].z)
                        throw Core::Exception("aabbMax error at node " + Utils::Convert::ToString(i + activeIndex) +
                                              ": CPU max " + Utils::CUDA::Convert::ToString(finalMax[i])
                                              + ", GPU max " + Utils::CUDA::Convert::ToString(gpuMax[i]));
                }

                delete finalMin;
                delete finalMax;
            }

            void TriangleMapUpperCreator::CreateChildren(int activeIndex, int activeRange,
                                                         int &childrenCreated){

                TriangleNode* nodes = map->GetNodes();
                int triangles = aabbMin->GetSize();

                unsigned int blocks = NextPow2(segments.GetSize()), threads = Segments::SEGMENT_SIZE;

                splitSide->Extend(triangles * 2, false);
                splitAddr->Extend(triangles * 2 + 1, false);
                leafSide->Extend(triangles * 2, false);
                leafAddr->Extend(triangles * 2 + 1, false);
                childSize->Extend(activeRange, false);
                nodes->Extend(nodes->GetSize() + activeRange * 2);

                SetSplitSide<<<blocks, threads>>>(segments.GetPrimitiveInfoData(),
                                                  segments.GetOwnerData(),
                                                  nodes->GetInfoData(),
                                                  nodes->GetSplitPositionData(),
                                                  aabbMin->GetDeviceData(),
                                                  aabbMax->GetDeviceData(),
                                                  splitSide->GetDeviceData());
                CHECK_FOR_CUDA_ERROR();

                cudppScan(scanHandle, splitAddr->GetDeviceData(), splitSide->GetDeviceData(), triangles * 2 + 1);

#ifdef CPU_VERIFY
                CheckSplits();
#endif

                int newTriangles;
                cudaMemcpy(&newTriangles, splitAddr->GetDeviceData() + triangles * 2, sizeof(int), cudaMemcpyDeviceToHost);
                //logger.info << "new triangles " << newTriangles << logger.end;
                CHECK_FOR_CUDA_ERROR();
                
                if (newTriangles < triangles)
                    throw Exception("New triangles amount " + Utils::Convert::ToString(newTriangles) + " was below old. WTF");
                
                bool createdLeafs = false;
                cudaMemcpyToSymbol(d_createdLeafs, &createdLeafs, sizeof(bool));

                unsigned int hatte, traade;
                Calc1DKernelDimensions(activeRange, hatte, traade);
                CalcNodeChildSize<<<hatte, traade>>>(nodes->GetPrimitiveInfoData() + activeIndex,
                                                     splitAddr->GetDeviceData(),
                                                     childSize->GetDeviceData());
                CHECK_FOR_CUDA_ERROR();
                cudaMemcpyFromSymbol(&createdLeafs, d_createdLeafs, sizeof(bool));

                /**
                 * if any empty nodes should be created!
                 * Create the empty nodes.
                 * Advance active index by the amount of empty nodes
                 *
                 * @OPT would it be advantageous to only run the
                 * kernel on the nodes actually producing empty
                 * splits? Or does it require to much computing power
                 * to calculate that. (Remember that warps with no
                 * splits can terminate pretty damn fast)
                 */
                /*
                bool emptySpaceSplitting = emptySpaceAddrs->GetSize() != 0;
                if (emptySpaceSplitting){
                    nodeIndices->Resize(activeRange, false);

                    int emptyNodes;
                    cudaMemcpy(&emptyNodes, emptySpaceAddrs->GetDeviceData() + emptySpaceAddrs->GetSize()-1, sizeof(int), cudaMemcpyDeviceToHost);
                    CHECK_FOR_CUDA_ERROR();

                    nodes->Resize(nodes->GetSize() + emptyNodes);

                    logger.info << "empty nodes: " << emptyNodes << logger.end;
                    logger.info << "nodes: " << nodes->GetSize() << logger.end;

                    logger.info << "emptySpacePlanes: " << Convert::ToString(emptySpacePlanes->GetDeviceData(), activeRange) << logger.end;
                    logger.info << "emptySpaceAddrs: " << Convert::ToString(emptySpaceAddrs->GetDeviceData(), activeRange) << logger.end;

                    Calc1DKernelDimensions(activeRange, hatte, traade);
                    CreateEmptyNodes<<<hatte, traade>>>(nodes->GetInfoData(), nodes->GetSplitPositionData(),
                                                        nodes->GetAabbMinData(), nodes->GetAabbMaxData(), 
                                                        nodes->GetPrimitiveInfoData(), nodes->GetChildrenData(),
                                                        emptySpacePlanes->GetDeviceData(), emptySpaceAddrs->GetDeviceData(),
                                                        nodeIndices->GetDeviceData());
                    CHECK_FOR_CUDA_ERROR();

                    logger.info << "nodeIndeces: " << Convert::ToString(nodeIndices->GetDeviceData(), activeRange) << logger.end;

                    for (int i = activeIndex; i < activeIndex + activeRange + emptyNodes; i++)
                        logger.info << nodes->ToString(i) << logger.end;
                    
                    activeIndex += emptyNodes;
                    cudaMemcpyToSymbol(d_activeNodeIndex, &activeIndex, sizeof(int));
                }else
                    nodeIndices->Resize(0, false);
                */

                if (createdLeafs){
                    logger.info << "Created leafs. Split resulted in " << newTriangles << " triangles." << logger.end;

                    SetPrimitiveLeafSide<<<blocks, threads>>>(segments.GetPrimitiveInfoData(),
                                                              segments.GetOwnerData(),
                                                              childSize->GetDeviceData(),
                                                              splitSide->GetDeviceData(),
                                                              leafSide->GetDeviceData());
                    CHECK_FOR_CUDA_ERROR();
                    
                    cudppScan(scanHandle, leafAddr->GetDeviceData(), leafSide->GetDeviceData(), triangles * 2 + 1);

                    int leafTriangles;
                    cudaMemcpy(&leafTriangles, leafAddr->GetDeviceData() + triangles * 2, sizeof(int), cudaMemcpyDeviceToHost);
                    
                    newTriangles -= leafTriangles;

                    tempAabbMin->Extend(newTriangles);
                    tempAabbMax->Extend(newTriangles);
                    int upperLeafPrimitives = primMax->GetSize();
                    primMax->Extend(upperLeafPrimitives + leafTriangles);
                    primMin->Extend(upperLeafPrimitives + leafTriangles);

                    SplitTriangles<<<blocks, threads>>>(segments.GetPrimitiveInfoData(),
                                                        segments.GetOwnerData(),
                                                        nodes->GetInfoData(),
                                                        nodes->GetSplitPositionData(),
                                                        splitSide->GetDeviceData(),
                                                        splitAddr->GetDeviceData(),
                                                        leafSide->GetDeviceData(),
                                                        leafAddr->GetDeviceData(),
                                                        aabbMin->GetDeviceData(),
                                                        aabbMax->GetDeviceData(),
                                                        tempAabbMin->GetDeviceData(),
                                                        tempAabbMax->GetDeviceData(),
                                                        primMin->GetDeviceData() + upperLeafPrimitives,
                                                        primMax->GetDeviceData() + upperLeafPrimitives);
                    CHECK_FOR_CUDA_ERROR();
                    std::swap(aabbMin, tempAabbMin);
                    std::swap(aabbMax, tempAabbMax);
                    
                    MarkNodeLeafs<<<hatte, traade>>>(childSize->GetDeviceData(),
                                                     leafSide->GetDeviceData());
                    CHECK_FOR_CUDA_ERROR();
                    
                    cudppScan(scanHandle, splitSide->GetDeviceData(), leafSide->GetDeviceData(), activeRange * 2 + 1);
                    CHECK_FOR_CUDA_ERROR();
                    
                    int leafNodes;
                    cudaMemcpy(&leafNodes, splitSide->GetDeviceData() + activeRange * 2, sizeof(int), cudaMemcpyDeviceToHost);
                    cudaMemcpyToSymbol(d_leafNodes, splitSide->GetDeviceData() + activeRange * 2, sizeof(int), 0, cudaMemcpyDeviceToDevice);

                    CreateUpperChildren<false>
                        <<<hatte, traade>>>(NULL, nodes->GetPrimitiveInfoData(),
                                            childSize->GetDeviceData(),
                                            splitAddr->GetDeviceData(),
                                            leafAddr->GetDeviceData(),
                                            splitSide->GetDeviceData(),
                                            nodes->GetChildrenData(),
                                            nodes->GetParentData(),
                                            upperLeafPrimitives);
                    CHECK_FOR_CUDA_ERROR();

                    childrenCreated = activeRange * 2 - leafNodes;

                    int upperNodeLeafs = leafIDs->GetSize();
                    leafIDs->Extend(leafIDs->GetSize() + leafNodes);
                    Calc1DKernelDimensions(leafNodes, blocks, threads);
                    int leafIndex = nodes->GetSize() - activeRange * 2;
                    MarkLeafNodes
                        <<<blocks, threads>>>(leafIDs->GetDeviceData() + upperNodeLeafs, 
                                              nodes->GetInfoData() + leafIndex,
                                              leafIndex, leafNodes);

                }else{
                    logger.info << "No leafs created. Split resulted in " << newTriangles << " triangles."  << logger.end;

                    tempAabbMin->Extend(newTriangles);
                    tempAabbMax->Extend(newTriangles);

                    CreateUpperChildren<false>
                        <<<hatte, traade>>>(NULL, nodes->GetPrimitiveInfoData(),
                                            childSize->GetDeviceData(),
                                            splitAddr->GetDeviceData(),
                                            nodes->GetChildrenData(),
                                            nodes->GetParentData());
                    CHECK_FOR_CUDA_ERROR();

                    SplitTriangles<<<blocks, threads>>>(segments.GetPrimitiveInfoData(),
                                                        segments.GetOwnerData(),
                                                        nodes->GetInfoData(),
                                                        nodes->GetSplitPositionData(),
                                                        splitSide->GetDeviceData(),
                                                        splitAddr->GetDeviceData(),
                                                        aabbMin->GetDeviceData(),
                                                        aabbMax->GetDeviceData(),
                                                        tempAabbMin->GetDeviceData(),
                                                        tempAabbMax->GetDeviceData());
                    CHECK_FOR_CUDA_ERROR();

                    std::swap(aabbMin, tempAabbMin);
                    std::swap(aabbMax, tempAabbMax);
                    
                    childrenCreated = activeRange * 2;
                }

                Calc1DKernelDimensions(activeRange, blocks, threads);
                PropagateChildAabb<false><<<blocks, threads>>>(NULL, nodes->GetInfoData(),
                                                               nodes->GetSplitPositionData(),
                                                               nodes->GetAabbMinData(),
                                                               nodes->GetAabbMaxData(),
                                                               nodes->GetChildrenData());
                CHECK_FOR_CUDA_ERROR();
                
#if CPU_VERIFY
                // Check that all primitive bounding boxes are tight or inside the primitive
                CheckPrimAabb(aabbMin, aabbMax);
                
                // Check that the nodes aabb cover all their respective primitives.
                for (int i = activeIndex; i < activeIndex + activeRange; ++i){
                    float4 parentAabbMin, parentAabbMax;
                    cudaMemcpy(&parentAabbMin, nodes->GetAabbMinData() + i, sizeof(float4), cudaMemcpyDeviceToHost);
                    cudaMemcpy(&parentAabbMax, nodes->GetAabbMaxData() + i, sizeof(float4), cudaMemcpyDeviceToHost);

                    CheckUpperNode(i, parentAabbMin, parentAabbMax, activeRange);
                }
#endif
            }
            
            void TriangleMapUpperCreator::CheckPrimAabb(CUDADataBlock<1, float4> *aabbMin, 
                                                        CUDADataBlock<1, float4> *aabbMax){
                int triangles = aabbMax->GetSize();
                GeometryList* geom = map->GetGeometry();

                float4 primAabbMin[triangles];
                cudaMemcpy(primAabbMin, aabbMin->GetDeviceData(), triangles * sizeof(float4), cudaMemcpyDeviceToHost);
                float4 primAabbMax[triangles];
                cudaMemcpy(primAabbMax, aabbMax->GetDeviceData(), triangles * sizeof(float4), cudaMemcpyDeviceToHost);
                CHECK_FOR_CUDA_ERROR();
                for (int i = 0; i < triangles; ++i){
                    int index = primAabbMin[i].w;
                    float4 p0, p1, p2;
                    cudaMemcpy(&p0, geom->p0->GetDeviceData() + index, sizeof(float4), cudaMemcpyDeviceToHost);
                    cudaMemcpy(&p1, geom->p1->GetDeviceData() + index, sizeof(float4), cudaMemcpyDeviceToHost);
                    cudaMemcpy(&p2, geom->p2->GetDeviceData() + index, sizeof(float4), cudaMemcpyDeviceToHost);

                    float4 aabbMin = min(p0, min(p1, p2));
                    float4 aabbMax = max(p0, max(p1, p2));

                    if (primAabbMin[i].x < aabbMin.x || 
                        primAabbMin[i].y < aabbMin.y || 
                        primAabbMin[i].z < aabbMin.z ||
                        aabbMax.x < primAabbMax[i].x ||
                        aabbMax.y < primAabbMax[i].y ||
                        aabbMax.z < primAabbMax[i].z)
                        throw Exception("Element " + Utils::Convert::ToString(i) + 
                                        " with cornors " + Convert::ToString(p0) +
                                        ", " + Convert::ToString(p1) + " and " + Convert::ToString(p2) +
                                        " is not strictly contained in aabb " + Convert::ToString(primAabbMin[i]) +
                                        " -> " + Convert::ToString(primAabbMax[i]));
                }
                CHECK_FOR_CUDA_ERROR();
            }
            
            void TriangleMapUpperCreator::CheckUpperNode(int index, float4 calcedAabbMin, 
                                                         float4 calcedAabbMax, int activeRange){
                //logger.info << "Checking node " << index << logger.end;
                char axis;
                cudaMemcpy(&axis, map->nodes->GetInfoData() + index, sizeof(char), cudaMemcpyDeviceToHost);
                
                if (axis == KDNode::LEAF){
                    CheckUpperLeaf(index, calcedAabbMin, calcedAabbMax);                    
                }else{
                    float splitPos;
                    cudaMemcpy(&splitPos, map->nodes->GetSplitPositionData() + index, sizeof(float), cudaMemcpyDeviceToHost);

                    int2 childrenIndex;
                    cudaMemcpy(&childrenIndex, map->nodes->GetChildrenData() + index, sizeof(int2), cudaMemcpyDeviceToHost);
                    
                    int leftIndex = childrenIndex.x;

                    /*
                    int leftParent;
                    cudaMemcpy(&leftParent, nodes->GetParentData() + leftIndex, sizeof(int), cudaMemcpyDeviceToHost);
                    if (leftParent != index)
                        throw Exception("Node " + Utils::Convert::ToString(leftIndex) +
                                        "'s parent " + Utils::Convert::ToString(leftParent) +
                                        " does not match actual parent " + Utils::Convert::ToString(index));
                    */

                    float4 leftAabbMin = calcedAabbMin;
                    float4 leftAabbMax = make_float4(axis == KDNode::X ? splitPos : calcedAabbMax.x,
                                                     axis == KDNode::Y ? splitPos : calcedAabbMax.y,
                                                     axis == KDNode::Z ? splitPos : calcedAabbMax.z,
                                                     calcedAabbMax.w);

                    if (leftIndex < map->nodes->GetSize() - 2 * activeRange)
                        CheckUpperNode(leftIndex, leftAabbMin, leftAabbMax, activeRange);
                    else
                        CheckUpperLeaf(leftIndex, leftAabbMin, leftAabbMax);

                    int rightIndex = childrenIndex.y;

                    /*                        
                    int rightParent;
                    cudaMemcpy(&rightParent, nodes->GetParentData() + rightIndex, sizeof(int), cudaMemcpyDeviceToHost);
                    if (rightParent != index)
                        throw Exception("Node " + Utils::Convert::ToString(rightIndex) +
                                        "'s parent " + Utils::Convert::ToString(rightParent) +
                                        " does not match actual parent " + Utils::Convert::ToString(index));
                    */
                        
                    float4 rightAabbMin = make_float4(axis == KDNode::X ? splitPos : calcedAabbMin.x,
                                                      axis == KDNode::Y ? splitPos : calcedAabbMin.y,
                                                      axis == KDNode::Z ? splitPos : calcedAabbMin.z,
                                                      calcedAabbMin.w);
                    float4 rightAabbMax = calcedAabbMax;

                    if (rightIndex < map->nodes->GetSize() - 2 * activeRange)
                        CheckUpperNode(rightIndex, rightAabbMin, rightAabbMax, activeRange);
                    else
                        CheckUpperLeaf(rightIndex, rightAabbMin, rightAabbMax);
                }                
            }

            void TriangleMapUpperCreator::CheckUpperLeaf(int index, float4 calcedAabbMin, float4 calcedAabbMax){
                //logger.info << "Node " << index << " is a leaf" << logger.end;
                int2 primInfo;
                cudaMemcpy(&primInfo, map->nodes->GetPrimitiveInfoData() + index, sizeof(int2), cudaMemcpyDeviceToHost);
                CHECK_FOR_CUDA_ERROR();
                
                bool isLeaf = primInfo.y < TriangleNode::MAX_LOWER_SIZE;
                for (int j = primInfo.x; j < primInfo.x + primInfo.y; ++j){
                    float4 h_primMin, h_primMax;
                    if (isLeaf){
                        cudaMemcpy(&h_primMin, primMin->GetDeviceData() + j, sizeof(float4), cudaMemcpyDeviceToHost);
                        cudaMemcpy(&h_primMax, primMax->GetDeviceData() + j, sizeof(float4), cudaMemcpyDeviceToHost);
                    }else{
                        cudaMemcpy(&h_primMin, aabbMin->GetDeviceData() + j, sizeof(float4), cudaMemcpyDeviceToHost);
                        cudaMemcpy(&h_primMax, aabbMax->GetDeviceData() + j, sizeof(float4), cudaMemcpyDeviceToHost);
                    }
                    CHECK_FOR_CUDA_ERROR();
                            
                    if (!aabbContains(calcedAabbMin, calcedAabbMax, h_primMin))
                        throw Core::Exception("primitive  " + Utils::Convert::ToString(j) + 
                                              "'s min " + Convert::ToString(h_primMin) +
                                              " not included in node " + Utils::Convert::ToString(index) +
                                              "'s aabb " + Convert::ToString(calcedAabbMin) +
                                              " -> " + Convert::ToString(calcedAabbMax));

                    if (!aabbContains(calcedAabbMin, calcedAabbMax, h_primMax))
                        throw Core::Exception("primitive  " + Utils::Convert::ToString(j) + 
                                              "'s max " + Convert::ToString(h_primMax) +
                                              " not included in node " + Utils::Convert::ToString(index) +
                                              "'s aabb " + Convert::ToString(calcedAabbMin) +
                                              " -> " + Convert::ToString(calcedAabbMax));
                }
            }

            void TriangleMapUpperCreator::CheckSplits() {
                int triangles = splitSide->GetSize() / 2;

                int sides[triangles * 2];
                cudaMemcpy(sides, splitSide->GetDeviceData(), triangles * 2 * sizeof(int), cudaMemcpyDeviceToHost);
                CHECK_FOR_CUDA_ERROR();

                int addrs[triangles * 2];
                cudaMemcpy(addrs, splitAddr->GetDeviceData(), (triangles * 2 + 1) * sizeof(int), cudaMemcpyDeviceToHost);
                CHECK_FOR_CUDA_ERROR();

                for (int i = 0; i < triangles; ++i){
                    // Check that a bounding box is at least assigned to one side.
                    if (sides[i] + sides[triangles + i] == 0)
                        throw Exception("Bounding box " + Utils::Convert::ToString(i) +
                                        "was neither left nor right.");
                }

                int prims = 0;
                for (int i = 1; i < triangles * 2 + 1; ++i){
                    prims += sides[i-1];
                    if (prims != addrs[i])
                        throw Exception("Stuff went wrong at bounding box " + Utils::Convert::ToString(i));
                }

                //logger.info << "New prims " << prims << logger.end;
            }

        }
    }
}
