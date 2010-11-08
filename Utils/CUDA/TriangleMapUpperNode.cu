// Triangle map class for CUDA
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

#include <Utils/CUDA/Kernels/PhotonMapDeviceVars.h>
#include <Utils/CUDA/Kernels/TriangleUpper.h>
#include <Utils/CUDA/Kernels/TriangleUpperSegment.h>
#include <Utils/CUDA/Kernels/ReduceSegments.h>
#include <Utils/CUDA/Kernels/TriangleUpperChildren.h>
#include <Utils/CUDA/Kernels/TriangleKernels.h>

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {

            using namespace Kernels;

            void TriangleMap::CreateUpperNodes(){
                int activeIndex = 0, activeRange = 1;
                int childrenCreated;
                
                upperNodeLeafs = upperLeafPrimitives = 0;
                
                cudaMemcpyToSymbol(d_emptySpaceThreshold, &emptySpaceThreshold, sizeof(float));

                // Setup root node!
                int2 i = make_int2(0, triangles);
                cudaMemcpy(nodes->GetPrimitiveInfoData(), &i, sizeof(int2), cudaMemcpyHostToDevice);
                int parent = 0;
                cudaMemcpy(nodes->GetParentData(), &parent, sizeof(int), cudaMemcpyHostToDevice);
                nodes->size = 1;

                // Setup bounding box info
                unsigned int blocks, threads;
                Calc1DKernelDimensions(triangles, blocks, threads);
                AddIndexToAabb<<<blocks, threads>>>(geom->GetAabbMinData(), geom->GetSize(), aabbMin->GetDeviceData());
                // @OPT Just switch the arrays.
                cudaMemcpy(aabbMax->GetDeviceData(), geom->GetAabbMaxData(), 
                           triangles * sizeof(float4), cudaMemcpyDeviceToDevice);
                CHECK_FOR_CUDA_ERROR();                

                START_TIMER(timerID);
                while (activeRange > 0){
                    ProcessUpperNodes(activeIndex, activeRange, 
                                      childrenCreated);

                    //for (int i = 0; i < activeRange; ++i)
                    //logger.info << nodes->ToString(i + activeIndex) << logger.end;

                    activeIndex = nodes->size - childrenCreated;
                    activeRange = childrenCreated;
                }
                PRINT_TIMER(timerID, "triangle upper map");

                triangles = resultMin->GetSize();

                Calc1DKernelDimensions(resultMin->GetSize(), blocks, threads, 128);
                START_TIMER(timerID);                
                AdjustBoundingBox<<<blocks, threads>>>(resultMin->GetDeviceData(), 
                                                       resultMax->GetDeviceData(),
                                                       geom->GetP0Data(),
                                                       geom->GetP1Data(),
                                                       geom->GetP2Data(),
                                                       geom->GetAabbMinData(),
                                                       geom->GetAabbMaxData(),
                                                       geom->GetSurfaceAreaData(),
                                                       resultMin->GetSize());
                PRINT_TIMER(timerID, "Adjusting bounding box");
                CHECK_FOR_CUDA_ERROR();

                /*
                float4 min[resultMin->GetSize()];
                cudaMemcpy(min, resultMin->GetDeviceData(), sizeof(float4) * resultMin->GetSize(), cudaMemcpyDeviceToHost);
                float4 max[resultMax->GetSize()];
                cudaMemcpy(max, resultMax->GetDeviceData(), sizeof(float4) * resultMax->GetSize(), cudaMemcpyDeviceToHost);
                int cnt = 0;
                for (int i = 0; i < resultMax->GetSize(); ++i)
                    if (max[i].w == 0.0f) cnt++;

                logger.info << "empty bb's " << cnt << logger.end;

                float3 aabbMin = make_float3(3, 0, -0.00001f);
                float3 aabbMax = make_float3(8, 1, 4);
                bool hit = TightTriangleBB(make_float3(0, 6, 0), make_float3(8, 4, 0), make_float3(6, 0, 0),
                                           aabbMin, aabbMax, true);

                if (hit)
                    logger.info << "min: " << Convert::ToString(aabbMin) << ", max: " << Convert::ToString(aabbMax) << logger.end;
                else
                    logger.info << "missed it: min: " << Convert::ToString(aabbMin) << ", max: " << Convert::ToString(aabbMax) << logger.end;
                */
            }

            void TriangleMap::ProcessUpperNodes(int activeIndex, int activeRange, 
                                                int &childrenCreated){

                //logger.info << "=== Process " << activeRange << " Upper Nodes Starting at " << activeIndex << " === with " << triangles << " triangles" << logger.end;

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

            void TriangleMap::Segment(int activeIndex, int activeRange){
                nodeSegments->Extend(activeRange+1);

                unsigned int blocks, threads;
                Calc1DKernelDimensions(activeRange, blocks, threads);
                NodeSegments<<<blocks, threads>>>(nodes->GetPrimitiveInfoData() + activeIndex,
                                                  nodeSegments->GetDeviceData());

                CHECK_FOR_CUDA_ERROR();
                cudppScan(scanHandle, nodeSegments->GetDeviceData(), nodeSegments->GetDeviceData(), activeRange+1);

                int amountOfSegments;
                cudaMemcpy(&amountOfSegments, nodeSegments->GetDeviceData() + activeRange, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpyToSymbol(d_segments, nodeSegments->GetDeviceData() + activeRange, sizeof(int), 0, cudaMemcpyDeviceToDevice);
                CHECK_FOR_CUDA_ERROR();

                segments.Extend(amountOfSegments);
                segments.size = amountOfSegments;

                cudaMemset(segments.GetOwnerData(), 0, amountOfSegments * sizeof(int));
                MarkOwnerStart<<<blocks, threads>>>(segments.GetOwnerData(),
                                                    nodeSegments->GetDeviceData());
                CHECK_FOR_CUDA_ERROR();

                cudppScan(scanInclHandle, segments.GetOwnerData(), segments.GetOwnerData(), amountOfSegments);

                Calc1DKernelDimensions(amountOfSegments, blocks, threads);
                CalcSegmentPrimitives<<<blocks, threads>>>(segments.GetOwnerData(),
                                                           nodeSegments->GetDeviceData(),
                                                           nodes->GetPrimitiveInfoData(),
                                                           segments.GetPrimitiveInfoData());
                CHECK_FOR_CUDA_ERROR();
            }

            void TriangleMap::ReduceAabb(int activeIndex, int activeRange){
                
                // Reduce aabb pr segment
                unsigned int blocks = segments.size;
                unsigned int threads = Segments::SEGMENT_SIZE;
                unsigned int memSize = 2 * sizeof(float4) * segments.SEGMENT_SIZE;

                //START_TIMER(timerID);
                logger.info << "ReduceSegments<<<" << blocks << ", " << threads << ", " << memSize << ">>>" << logger.end;
                ReduceSegmentsShared<<<blocks, threads, memSize>>>(segments.GetPrimitiveInfoData(),
                                                                   aabbMin->GetDeviceData(), aabbMax->GetDeviceData(),
                                                                   segments.GetAabbMinData(), segments.GetAabbMaxData());
                //PRINT_TIMER(timerID, "ReduceSegments");
                // @TODO has provoked an "unspecified launch failure"
                CHECK_FOR_CUDA_ERROR();

#if CPU_VERIFY
                int2 info[segments.size];
                cudaMemcpy(info, segments.GetPrimitiveInfoData(), 
                           segments.size * sizeof(int2), cudaMemcpyDeviceToHost);

                float4 segMin[segments.size];
                cudaMemcpy(segMin, segments.GetAabbMinData(), 
                           segments.size * sizeof(float4), cudaMemcpyDeviceToHost);
                float4 segMax[segments.size];
                cudaMemcpy(segMax, segments.GetAabbMaxData(), 
                           segments.size * sizeof(float4), cudaMemcpyDeviceToHost);

                for (int i = 0; i < segments.size; ++i){
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

                // Do the final reduce
                int segOwner[segments.size];
                cudaMemcpy(segOwner, segments.GetOwnerData(), 
                           segments.size * sizeof(int), cudaMemcpyDeviceToHost);
                
                float4 cpuMin[activeRange];
                float4 cpuMax[activeRange];

                int owner0 = segOwner[0];
                float4 localMin = segMin[0];
                float4 localMax = segMax[0];
                for (int i = 1; i < segments.size; ++i){
                    int owner1 = segOwner[i];
                    if (owner0 != owner1){
                        cpuMin[owner0 - activeIndex] = localMin;
                        cpuMax[owner0 - activeIndex] = localMax;
                        owner0 = segOwner[i];
                        localMin = segMin[i];
                        localMax = segMax[i];
                    }else{
                        localMin = min(localMin, segMin[i]);
                        localMax = max(localMax, segMax[i]);
                    }
                }
                cpuMin[owner0 - activeIndex] = localMin;
                cpuMax[owner0 - activeIndex] = localMax;

#endif

                //threads = min(blocks, activeCudaDevice.maxThreadsDim[0]);
                threads = min((segments.size / 32) * 32 + 32, activeCudaDevice.maxThreadsDim[0]);
                //START_TIMER(timerID);
                logger.info << "SegmentedReduce0<<<1, " << threads << ">>>" << logger.end;
                SegmentedReduce0<<<1, threads>>>(segments.GetAabbMinData(),
                                                 segments.GetAabbMaxData(),
                                                 segments.GetOwnerData(),
                                                 nodes->GetAabbMinData(),
                                                 nodes->GetAabbMaxData());
                //PRINT_TIMER(timerID, "Segmented reduce");
                CHECK_FOR_CUDA_ERROR();

#if CPU_VERIFY
                float4 gpuMin[activeRange];
                cudaMemcpy(gpuMin, nodes->GetAabbMinData() + activeIndex,
                           activeRange * sizeof(float4), cudaMemcpyDeviceToHost);
                float4 gpuMax[activeRange];
                cudaMemcpy(gpuMax, nodes->GetAabbMaxData() + activeIndex,
                           activeRange * sizeof(float4), cudaMemcpyDeviceToHost);
                for (int i = 0; i < activeRange; ++i){
                    if (cpuMin[i].x != gpuMin[i].x || cpuMin[i].y != gpuMin[i].y || cpuMin[i].z != gpuMin[i].z)
                        throw Core::Exception("aabbMin error at node " + Utils::Convert::ToString(i + activeIndex) +
                                              ": CPU min " + Utils::CUDA::Convert::ToString(cpuMin[i])
                                              + ", GPU min " + Utils::CUDA::Convert::ToString(gpuMin[i]));

                    if (cpuMax[i].x != gpuMax[i].x || cpuMax[i].y != gpuMax[i].y || cpuMax[i].z != gpuMax[i].z)
                        throw Core::Exception("aabbMax error at node " + Utils::Convert::ToString(i + activeIndex) +
                                              ": CPU max " + Utils::CUDA::Convert::ToString(cpuMax[i])
                                              + ", GPU max " + Utils::CUDA::Convert::ToString(gpuMax[i]));
                }
#endif

                // Calc splitting planes.
                Calc1DKernelDimensions(activeRange, blocks, threads);
                CalcUpperNodeSplitInfo<<<blocks, threads>>>(nodes->GetAabbMinData() + activeIndex,
                                                            nodes->GetAabbMaxData() + activeIndex,
                                                            nodes->GetSplitPositionData() + activeIndex,
                                                            nodes->GetInfoData() + activeIndex);
                CHECK_FOR_CUDA_ERROR();
            }

            void TriangleMap::CreateChildren(int activeIndex, int activeRange,
                                             int &childrenCreated){
                unsigned int blocks = NextPow2(segments.size), threads = Segments::SEGMENT_SIZE;

                splitSide->Extend(triangles * 2, false);
                splitAddr->Extend(triangles * 2 + 1, false);
                leafSide->Extend(triangles * 2, false);
                leafAddr->Extend(triangles * 2 + 1, false);
                childSize->Extend(activeRange, false);
                nodes->Extend(nodes->size + activeRange * 2);

                //START_TIMER(timerID);
                SetSplitSide<<<blocks, threads>>>(segments.GetPrimitiveInfoData(),
                                                  segments.GetOwnerData(),
                                                  nodes->GetInfoData(),
                                                  nodes->GetSplitPositionData(),
                                                  aabbMin->GetDeviceData(),
                                                  aabbMax->GetDeviceData(),
                                                  splitSide->GetDeviceData());
                //PRINT_TIMER(timerID, "Set split Side");
                CHECK_FOR_CUDA_ERROR();

                cudppScan(scanHandle, splitAddr->GetDeviceData(), splitSide->GetDeviceData(), triangles * 2 + 1);
                //logger.info << "splitAddr " << Convert::ToString(splitAddr->GetDeviceData(), 20) << logger.end;

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

                /*
                EmptySpaceSplits<<<hatte, traade>>>(nodes->GetAabbMinData(),
                                                    nodes->GetAabbMaxData(),
                                                    nodes->GetInfoData(),
                                                    nodes->GetSplitPositionData(),
                                                    nodes->GetParentData(),
                                                    emptySpaceSplits->GetDeviceData());
                CHECK_FOR_CUDA_ERROR();
                cudppScan(scanHandle, emptySpaceAddrs->GetDeviceData(), emptySpaceSplits->GetDeviceData(), activeRange+1);

                logger.info << "Empty space splits: " << Convert::ToString(emptySpaceSplits->GetDeviceData(), activeRange) << logger.end;
                */

                if (createdLeafs){
                    //logger.info << "Created leafs" << logger.end;

                    SetPrimitiveLeafSide<<<blocks, threads>>>(segments.GetPrimitiveInfoData(),
                                                              segments.GetOwnerData(),
                                                              childSize->GetDeviceData(),
                                                              splitSide->GetDeviceData(),
                                                              leafSide->GetDeviceData());
                    CHECK_FOR_CUDA_ERROR();
                    
                    cudppScan(scanHandle, leafAddr->GetDeviceData(), leafSide->GetDeviceData(), triangles * 2 + 1);

                    /*
                    logger.info << "SetPrimitiveLeafSide<<<" << blocks << ", " << Segments::SEGMENT_SIZE << ">>>" << logger.end;
                    logger.info << "Segments primitive info: " << Utils::CUDA::Convert::ToString(segments.GetPrimitiveInfoData(), segments.size) << logger.end;
                    logger.info << "Segments owner: " << Utils::CUDA::Convert::ToString(segments.GetOwnerData(), segments.size) << logger.end;
                    logger.info << "Child sizes: " << Utils::CUDA::Convert::ToString(childSize->GetDeviceData(), activeRange) << logger.end;
                    logger.info << "Split side: " << Utils::CUDA::Convert::ToString(splitSide->GetDeviceData()+triangles-100, 200) << logger.end;
                    logger.info << "===" << logger.end;
                    logger.info << "Leaf side: " << Utils::CUDA::Convert::ToString(leafSide->GetDeviceData()+triangles-100, 200) << logger.end;
                    logger.info << "Leaf addr: " << Utils::CUDA::Convert::ToString(leafAddr->GetDeviceData()+triangles-100, 200) << logger.end;
                    */

                    int leafTriangles;
                    cudaMemcpy(&leafTriangles, leafAddr->GetDeviceData() + triangles * 2, sizeof(int), cudaMemcpyDeviceToHost);
                    //logger.info << "leaf triangles: " << leafTriangles << logger.end;
                    
                    newTriangles -= leafTriangles;

                    tempAabbMin->Extend(newTriangles);
                    tempAabbMax->Extend(newTriangles);
                    resultMax->Extend(upperLeafPrimitives + leafTriangles);
                    resultMin->Extend(upperLeafPrimitives + leafTriangles);

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
                                                        resultMin->GetDeviceData() + upperLeafPrimitives,
                                                        resultMax->GetDeviceData() + upperLeafPrimitives);
                    CHECK_FOR_CUDA_ERROR();
                    std::swap(aabbMin, tempAabbMin);
                    std::swap(aabbMax, tempAabbMax);
                    
                    // @TODO handle leaf nodes, probably add indices
                    // to an array for future processing.

                    MarkNodeLeafs<<<hatte, traade>>>(childSize->GetDeviceData(),
                                                     leafSide->GetDeviceData());
                    CHECK_FOR_CUDA_ERROR();
                    cudppScan(scanHandle, splitSide->GetDeviceData(), leafSide->GetDeviceData(), activeRange * 2 + 1);
                    CHECK_FOR_CUDA_ERROR();
                    
                    int leafNodes;
                    cudaMemcpy(&leafNodes, splitSide->GetDeviceData() + activeRange * 2, sizeof(int), cudaMemcpyDeviceToHost);
                    cudaMemcpyToSymbol(d_leafNodes, splitSide->GetDeviceData() + activeRange * 2, sizeof(int), 0, cudaMemcpyDeviceToDevice);
                    //logger.info << "leaf nodes: " << leafNodes << logger.end;

                    /*                                        
                    logger.info << "CreateChildren<<<" << hatte << ", " << traade << ">>>" << logger.end;
                    logger.info << "primitive info " << Convert::ToString(nodes->GetPrimitiveInfoData() + activeIndex, activeRange) << logger.end;
                    logger.info << "child size " << Convert::ToString(childSize->GetDeviceData(), activeRange) << logger.end;
                    logger.info << "Node leaf addrs " << Convert::ToString(splitSide->GetDeviceData(), activeRange * 2 + 1) << logger.end;
                    */

                    Kernels::CreateChildren
                        <<<hatte, traade>>>(nodes->GetPrimitiveInfoData(),
                                            childSize->GetDeviceData(),
                                            splitAddr->GetDeviceData(),
                                            leafAddr->GetDeviceData(),
                                            splitSide->GetDeviceData(),
                                            nodes->GetLeftData(),
                                            nodes->GetRightData(),
                                            nodes->GetParentData(),
                                            upperLeafPrimitives);
                    CHECK_FOR_CUDA_ERROR();

                    upperLeafPrimitives += leafTriangles;
                    triangles = newTriangles;
                    nodes->size += activeRange * 2;
                    childrenCreated = activeRange * 2 - leafNodes;

                    upperNodeLeafList->Extend(upperNodeLeafs + leafNodes);
                    Calc1DKernelDimensions(leafNodes, blocks, threads);
                    int leafIndex = nodes->size - activeRange * 2;
                    //logger.info << "leaf index " << leafIndex << logger.end;
                    MarkLeafNodes
                        <<<blocks, threads>>>(upperNodeLeafList->GetDeviceData() + upperNodeLeafs, 
                                              nodes->GetInfoData() + leafIndex,
                                              leafIndex, leafNodes);
                    upperNodeLeafs += leafNodes;
                    
                    //logger.info << "UpperNode Leafs: " << upperNodeLeafs << logger.end;
                    
                }else{
                    logger.info << "No leafs created. Split resulted in " << newTriangles << " triangles."  << logger.end;

                    tempAabbMin->Extend(newTriangles);
                    tempAabbMax->Extend(newTriangles);

                    Kernels::CreateChildren
                        <<<hatte, traade>>>(nodes->GetPrimitiveInfoData(),
                                            childSize->GetDeviceData(),
                                            splitAddr->GetDeviceData(),
                                            nodes->GetLeftData(),
                                            nodes->GetRightData(),
                                            nodes->GetParentData());
                    CHECK_FOR_CUDA_ERROR();

                    //logger.info << "Left " << Utils::CUDA::Convert::ToString(nodes->GetLeftData() + activeIndex, activeRange) << logger.end;
                    //logger.info << "Right " << Utils::CUDA::Convert::ToString(nodes->GetRightData() + activeIndex, activeRange) << logger.end;
                    //logger.info << "Children primitive info: " << Utils::CUDA::Convert::ToString(nodes->GetPrimitiveInfoData() + activeIndex + activeRange, activeRange * 2) << logger.end;

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
                    
                    nodes->size += activeRange * 2;
                    childrenCreated = activeRange * 2;
                    triangles = newTriangles;
                }

#if CPU_VERIFY
                // Check that all primitive bounding boxes are tight or inside the primitive
                
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

                // Check that the nodes aabb cover all their respective primitives.
                for (int i = activeIndex; i < activeIndex + activeRange; ++i){
                    float4 parentAabbMin, parentAabbMax;
                    cudaMemcpy(&parentAabbMin, nodes->GetAabbMinData() + i, sizeof(float4), cudaMemcpyDeviceToHost);
                    cudaMemcpy(&parentAabbMax, nodes->GetAabbMaxData() + i, sizeof(float4), cudaMemcpyDeviceToHost);

                    CheckUpperNode(i, parentAabbMin, parentAabbMax, activeRange);
                }
#endif

            }

            void TriangleMap::CheckUpperNode(int index, float4 calcedAabbMin, float4 calcedAabbMax, int activeRange){
                //logger.info << "Checking node " << index << logger.end;
                char axis;
                cudaMemcpy(&axis, nodes->GetInfoData() + index, sizeof(char), cudaMemcpyDeviceToHost);
                
                if (axis == KDNode::LEAF){
                    CheckUpperLeaf(index, calcedAabbMin, calcedAabbMax);                    
                }else{
                    float splitPos;
                    cudaMemcpy(&splitPos, nodes->GetSplitPositionData() + index, sizeof(float), cudaMemcpyDeviceToHost);
                    
                    int leftIndex;
                    cudaMemcpy(&leftIndex, nodes->GetLeftData() + index, sizeof(int), cudaMemcpyDeviceToHost);

                    int leftParent;
                    cudaMemcpy(&leftParent, nodes->GetParentData() + leftIndex, sizeof(int), cudaMemcpyDeviceToHost);
                        
                    if (leftParent != index)
                        throw Exception("Node " + Utils::Convert::ToString(leftIndex) +
                                        "'s parent " + Utils::Convert::ToString(leftParent) +
                                        " does not match actual parent " + Utils::Convert::ToString(index));

                    float4 leftAabbMin = calcedAabbMin;
                    float4 leftAabbMax = make_float4(axis == KDNode::X ? splitPos : calcedAabbMax.x,
                                                     axis == KDNode::Y ? splitPos : calcedAabbMax.y,
                                                     axis == KDNode::Z ? splitPos : calcedAabbMax.z,
                                                     calcedAabbMax.w);

                    if (leftIndex < nodes->size - 2 * activeRange)
                        CheckUpperNode(leftIndex, leftAabbMin, leftAabbMax, activeRange);
                    else
                        CheckUpperLeaf(leftIndex, leftAabbMin, leftAabbMax);

                    int rightIndex;
                    cudaMemcpy(&rightIndex, nodes->GetRightData() + index, sizeof(int), cudaMemcpyDeviceToHost);
                        
                    int rightParent;
                    cudaMemcpy(&rightParent, nodes->GetParentData() + rightIndex, sizeof(int), cudaMemcpyDeviceToHost);

                    if (rightParent != index)
                        throw Exception("Node " + Utils::Convert::ToString(rightIndex) +
                                        "'s parent " + Utils::Convert::ToString(rightParent) +
                                        " does not match actual parent " + Utils::Convert::ToString(index));
                        
                    float4 rightAabbMin = make_float4(axis == KDNode::X ? splitPos : calcedAabbMin.x,
                                                      axis == KDNode::Y ? splitPos : calcedAabbMin.y,
                                                      axis == KDNode::Z ? splitPos : calcedAabbMin.z,
                                                      calcedAabbMin.w);
                    float4 rightAabbMax = calcedAabbMax;

                    if (rightIndex < nodes->size - 2 * activeRange)
                        CheckUpperNode(rightIndex, rightAabbMin, rightAabbMax, activeRange);
                    else
                        CheckUpperLeaf(rightIndex, rightAabbMin, rightAabbMax);
                }                
            }

            void TriangleMap::CheckUpperLeaf(int index, float4 calcedAabbMin, float4 calcedAabbMax){
                //logger.info << "Node " << index << " is a leaf" << logger.end;
                int2 primInfo;
                cudaMemcpy(&primInfo, nodes->GetPrimitiveInfoData() + index, sizeof(int2), cudaMemcpyDeviceToHost);
                CHECK_FOR_CUDA_ERROR();
                
                bool isLeaf = primInfo.y < TriangleNode::MAX_LOWER_SIZE;
                for (int j = primInfo.x; j < primInfo.x + primInfo.y; ++j){
                    float4 primMin, primMax;
                    if (isLeaf){
                        cudaMemcpy(&primMin, resultMin->GetDeviceData() + j, sizeof(float4), cudaMemcpyDeviceToHost);
                        cudaMemcpy(&primMax, resultMax->GetDeviceData() + j, sizeof(float4), cudaMemcpyDeviceToHost);
                    }else{
                        cudaMemcpy(&primMin, aabbMin->GetDeviceData() + j, sizeof(float4), cudaMemcpyDeviceToHost);
                        cudaMemcpy(&primMax, aabbMax->GetDeviceData() + j, sizeof(float4), cudaMemcpyDeviceToHost);
                    }
                    CHECK_FOR_CUDA_ERROR();
                            
                    if (!aabbContains(calcedAabbMin, calcedAabbMax, primMin))
                        throw Core::Exception("primitive  " + Utils::Convert::ToString(j) + 
                                              "'s min " + Convert::ToString(primMin) +
                                              " not included in node " + Utils::Convert::ToString(index) +
                                              "'s aabb " + Convert::ToString(calcedAabbMin) +
                                              " -> " + Convert::ToString(calcedAabbMax));

                    if (!aabbContains(calcedAabbMin, calcedAabbMax, primMax))
                        throw Core::Exception("primitive  " + Utils::Convert::ToString(j) + 
                                              "'s max " + Convert::ToString(primMax) +
                                              " not included in node " + Utils::Convert::ToString(index) +
                                              "'s aabb " + Convert::ToString(calcedAabbMin) +
                                              " -> " + Convert::ToString(calcedAabbMax));
                }
            }

            void TriangleMap::CheckSplits(){
                int sides[triangles * 2];
                cudaMemcpy(sides, splitSide->GetDeviceData(), triangles * 2 * sizeof(int), cudaMemcpyDeviceToHost);
                CHECK_FOR_CUDA_ERROR();

                int addrs[triangles * 2];
                cudaMemcpy(addrs, splitAddr->GetDeviceData(), (triangles * 2 + 1) * sizeof(int), cudaMemcpyDeviceToHost);
                CHECK_FOR_CUDA_ERROR();

                //int prims = 0;
                for (int i = 0; i < triangles; ++i){
                    // Check that a bounding box is at least assigned to one side.
                    //prims += sides[i] + sides[triangles + i];
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
