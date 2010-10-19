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

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {

            using namespace Kernels;


            void TriangleMap::CreateUpperNodes(){
                int activeIndex = 0, activeRange = 1;
                int childrenCreated;
                
                upperNodeLeafs = upperLeafPrimitives = 0;

                // Setup root node!
                int2 i = make_int2(0, triangles);
                cudaMemcpy(upperNodes->GetPrimitiveInfoData(), &i, sizeof(int2), cudaMemcpyHostToDevice);
                upperNodes->size = 1;

                // Setup bounding box info
                unsigned int blocks, threads;
                Calc1DKernelDimensions(triangles, blocks, threads);
                AddIndexToAabb<<<blocks, threads>>>(geom->GetAabbMinData(), geom->GetSize(), aabbMin->GetDeviceData());
                // @OPT Just switch the arrays.
                cudaMemcpy(aabbMax->GetDeviceData(), geom->GetAabbMaxData(), 
                           triangles * sizeof(float4), cudaMemcpyDeviceToDevice);
                CHECK_FOR_CUDA_ERROR();                

                while (activeRange > 0){
                    ProcessUpperNodes(activeIndex, activeRange, 
                                      childrenCreated);

                    activeIndex = upperNodes->size - childrenCreated;
                    activeRange = childrenCreated;
                }

            }

            void TriangleMap::ProcessUpperNodes(int activeIndex, int activeRange, 
                                                int &childrenCreated){

                logger.info << "=== Process " << activeRange << " Upper Nodes Starting at " << activeIndex << " === with " << triangles << " triangles" << logger.end;

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
                if (nodeSegments->GetSize() < (unsigned int)activeRange+1)
                    nodeSegments->Resize(activeRange+1);

                unsigned int blocks, threads;
                Calc1DKernelDimensions(activeRange, blocks, threads);
                NodeSegments<<<blocks, threads>>>(upperNodes->GetPrimitiveInfoData() + activeIndex,
                                                  nodeSegments->GetDeviceData());

                CHECK_FOR_CUDA_ERROR();
                cudppScan(scanHandle, nodeSegments->GetDeviceData(), nodeSegments->GetDeviceData(), activeRange+1);

                int amountOfSegments;
                cudaMemcpy(&amountOfSegments, nodeSegments->GetDeviceData() + activeRange, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpyToSymbol(d_segments, nodeSegments->GetDeviceData() + activeRange, sizeof(int), 0, cudaMemcpyDeviceToDevice);
                CHECK_FOR_CUDA_ERROR();

                if (segments.maxSize < amountOfSegments)
                    segments.Resize(amountOfSegments);
                segments.size = amountOfSegments;

                cudaMemset(segments.GetOwnerData(), 0, amountOfSegments * sizeof(int));
                MarkOwnerStart<<<blocks, threads>>>(segments.GetOwnerData(),
                                                    nodeSegments->GetDeviceData());
                CHECK_FOR_CUDA_ERROR();

                cudppScan(scanInclHandle, segments.GetOwnerData(), segments.GetOwnerData(), amountOfSegments);

                Calc1DKernelDimensions(amountOfSegments, blocks, threads);
                CalcSegmentPrimitives<<<blocks, threads>>>(segments.GetOwnerData(),
                                                           nodeSegments->GetDeviceData(),
                                                           upperNodes->GetPrimitiveInfoData(),
                                                           segments.GetPrimitiveInfoData());
                CHECK_FOR_CUDA_ERROR();
            }

            void TriangleMap::ReduceAabb(int activeIndex, int activeRange){
                
                // Reduce aabb pr segment
                unsigned int blocks = NextPow2(segments.size);
                unsigned int threads = Segments::SEGMENT_SIZE;
                unsigned int memSize = 2 * sizeof(float4) * segments.SEGMENT_SIZE;

                //START_TIMER(timerID);
                ReduceSegments<<<blocks, threads, memSize>>>(segments.GetPrimitiveInfoData(),
                                                             aabbMin->GetDeviceData(), aabbMax->GetDeviceData(),
                                                             segments.GetAabbMinData(), segments.GetAabbMaxData());
                //PRINT_TIMER(timerID, "ReduceSegments");
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
                SegmentedReduce0<<<1, threads>>>(segments.GetAabbMinData(),
                                                 segments.GetAabbMaxData(),
                                                 segments.GetOwnerData(),
                                                 upperNodes->GetAabbMinData(),
                                                 upperNodes->GetAabbMaxData());
                //PRINT_TIMER(timerID, "Segmented reduce");
                CHECK_FOR_CUDA_ERROR();

                /*                
                if (activeIndex == 92){
                    for (int i = 0; i < segments.size; ++i)
                        logger.info << "Segment " << i << "'s owner is " << segOwner[i] << " with max " << Convert::ToString(segMax[i]) << logger.end;                        
                }
                */

#if CPU_VERIFY
                float4 gpuMin[activeRange];
                cudaMemcpy(gpuMin, upperNodes->GetAabbMinData() + activeIndex,
                           activeRange * sizeof(float4), cudaMemcpyDeviceToHost);
                float4 gpuMax[activeRange];
                cudaMemcpy(gpuMax, upperNodes->GetAabbMaxData() + activeIndex,
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
                CalcUpperNodeSplitInfo<<<blocks, threads>>>(upperNodes->GetAabbMinData() + activeIndex,
                                                            upperNodes->GetAabbMaxData() + activeIndex,
                                                            upperNodes->GetSplitPositionData() + activeIndex,
                                                            upperNodes->GetInfoData() + activeIndex);
                CHECK_FOR_CUDA_ERROR();
            }

            void TriangleMap::CreateChildren(int activeIndex, int activeRange,
                                             int &childrenCreated){
                unsigned int blocks = NextPow2(segments.size), threads = Segments::SEGMENT_SIZE;

                /*
                if (activeIndex > 0){
                    // Do empty space splitting and update activeIndex and activeRange
                    throw Exception("Empty space splitting not implemented.");
                }
                */

                if (splitSide->GetSize() < (unsigned int)triangles * 2) splitSide->Resize(triangles * 2, false);
                if (splitAddr->GetSize() < (unsigned int)triangles * 2 + 1) splitAddr->Resize(triangles * 2 + 1, false);
                if (leafSide->GetSize() < (unsigned int)triangles * 2) leafSide->Resize(triangles * 2, false);
                if (leafAddr->GetSize() < (unsigned int)triangles * 2 + 1) leafAddr->Resize(triangles * 2 + 1, false);
                if (childSize->GetSize() < (unsigned int)activeRange) childSize->Resize(activeRange, false);
                if (upperNodes->maxSize < upperNodes->size + activeRange * 2) upperNodes->Resize(upperNodes->size + activeRange * 2);

                //START_TIMER(timerID);
                SetSplitSide<<<blocks, threads>>>(segments.GetPrimitiveInfoData(),
                                                  segments.GetOwnerData(),
                                                  upperNodes->GetInfoData(),
                                                  upperNodes->GetSplitPositionData(),
                                                  aabbMin->GetDeviceData(),
                                                  aabbMax->GetDeviceData(),
                                                  splitSide->GetDeviceData());
                //PRINT_TIMER(timerID, "Set split Side");
                CHECK_FOR_CUDA_ERROR();

                cudppScan(scanHandle, splitAddr->GetDeviceData(), splitSide->GetDeviceData(), triangles * 2 + 1);

                int newTriangles;
                cudaMemcpy(&newTriangles, splitAddr->GetDeviceData() + triangles * 2, sizeof(int), cudaMemcpyDeviceToHost);
                //logger.info << "new triangles " << newTriangles << logger.end;
                
                bool createdLeafs = false;
                cudaMemcpyToSymbol(d_createdLeafs, &createdLeafs, sizeof(bool));

                unsigned int hatte, traade;
                Calc1DKernelDimensions(activeRange, hatte, traade);
                CalcNodeChildSize<<<hatte, traade>>>(upperNodes->GetPrimitiveInfoData() + activeIndex,
                                                   splitAddr->GetDeviceData(),
                                                   childSize->GetDeviceData());
                CHECK_FOR_CUDA_ERROR();
                cudaMemcpyFromSymbol(&createdLeafs, d_createdLeafs, sizeof(bool));

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

                    if (tempAabbMin->GetSize() < (unsigned int) newTriangles) tempAabbMin->Resize(newTriangles);
                    if (tempAabbMax->GetSize() < (unsigned int) newTriangles) tempAabbMax->Resize(newTriangles);
                    if (geom->GetAabbMin()->GetSize() < (unsigned int) upperLeafPrimitives + leafTriangles)
                        geom->GetAabbMin()->Resize(upperLeafPrimitives + leafTriangles);
                    if (geom->GetAabbMax()->GetSize() < (unsigned int) upperLeafPrimitives + leafTriangles)
                        geom->GetAabbMax()->Resize(upperLeafPrimitives + leafTriangles);

                    SplitTriangles<<<blocks, threads>>>(segments.GetPrimitiveInfoData(),
                                                        segments.GetOwnerData(),
                                                        upperNodes->GetInfoData(),
                                                        upperNodes->GetSplitPositionData(),
                                                        splitSide->GetDeviceData(),
                                                        splitAddr->GetDeviceData(),
                                                        leafSide->GetDeviceData(),
                                                        leafAddr->GetDeviceData(),
                                                        aabbMin->GetDeviceData(),
                                                        aabbMax->GetDeviceData(),
                                                        tempAabbMin->GetDeviceData(),
                                                        tempAabbMax->GetDeviceData(),
                                                        geom->GetAabbMinData() + upperLeafPrimitives,
                                                        geom->GetAabbMaxData() + upperLeafPrimitives);
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
                    //logger.info << "leaf nodes: " << leafNodes << logger.end;

                    /*                                        
                    logger.info << "CreateChildren<<<" << hatte << ", " << traade << ">>>" << logger.end;
                    logger.info << "primitive info " << Convert::ToString(upperNodes->GetPrimitiveInfoData() + activeIndex, activeRange) << logger.end;
                    logger.info << "child size " << Convert::ToString(childSize->GetDeviceData(), activeRange) << logger.end;
                    logger.info << "Node leaf addrs " << Convert::ToString(splitSide->GetDeviceData(), activeRange * 2 + 1) << logger.end;
                    */

                    Kernels::CreateChildren
                        <<<hatte, traade>>>(upperNodes->GetPrimitiveInfoData() + activeIndex,
                                            childSize->GetDeviceData(),
                                            splitAddr->GetDeviceData(),
                                            leafAddr->GetDeviceData(),
                                            splitSide->GetDeviceData(),
                                            upperNodes->GetLeftData() + activeIndex,
                                            upperNodes->GetRightData() + activeIndex,
                                            upperLeafPrimitives);
                    CHECK_FOR_CUDA_ERROR();

                    upperLeafPrimitives += leafTriangles;
                    triangles = newTriangles;
                    upperNodes->size += activeRange * 2;
                    childrenCreated = activeRange * 2 - leafNodes;

                    upperNodeLeafList->Extend(upperNodeLeafs + leafNodes);
                    Calc1DKernelDimensions(leafNodes, blocks, threads);
                    int leafIndex = upperNodes->size - activeRange * 2;
                    //logger.info << "leaf index " << leafIndex << logger.end;
                    MarkLeafNodes
                        <<<blocks, threads>>>(upperNodeLeafList->GetDeviceData() + upperNodeLeafs, 
                                              upperNodes->GetInfoData() + leafIndex,
                                              leafIndex, leafNodes);
                    upperNodeLeafs += leafNodes;
                    
                    //logger.info << "UpperNode Leafs: " << upperNodeLeafs << logger.end;
                    
                }else{
                    //logger.info << "No leafs created" << logger.end;

                    if (tempAabbMin->GetSize() < (unsigned int) newTriangles) tempAabbMin->Resize(newTriangles);
                    if (tempAabbMax->GetSize() < (unsigned int) newTriangles) tempAabbMax->Resize(newTriangles);

                    Kernels::CreateChildren
                        <<<hatte, traade>>>(upperNodes->GetPrimitiveInfoData() + activeIndex,
                                            childSize->GetDeviceData(),
                                            splitAddr->GetDeviceData(),
                                            upperNodes->GetLeftData() + activeIndex,
                                            upperNodes->GetRightData() + activeIndex);
                    CHECK_FOR_CUDA_ERROR();

                    //logger.info << "Left " << Utils::CUDA::Convert::ToString(upperNodes->GetLeftData() + activeIndex, activeRange) << logger.end;
                    //logger.info << "Right " << Utils::CUDA::Convert::ToString(upperNodes->GetRightData() + activeIndex, activeRange) << logger.end;
                    //logger.info << "Children primitive info: " << Utils::CUDA::Convert::ToString(upperNodes->GetPrimitiveInfoData() + activeIndex + activeRange, activeRange * 2) << logger.end;

                    SplitTriangles<<<blocks, threads>>>(segments.GetPrimitiveInfoData(),
                                                        segments.GetOwnerData(),
                                                        upperNodes->GetInfoData(),
                                                        upperNodes->GetSplitPositionData(),
                                                        splitSide->GetDeviceData(),
                                                        splitAddr->GetDeviceData(),
                                                        aabbMin->GetDeviceData(),
                                                        aabbMax->GetDeviceData(),
                                                        tempAabbMin->GetDeviceData(),
                                                        tempAabbMax->GetDeviceData());
                    CHECK_FOR_CUDA_ERROR();

                    std::swap(aabbMin, tempAabbMin);
                    std::swap(aabbMax, tempAabbMax);
                    
                    upperNodes->size += activeRange * 2;
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
                    char axis;
                    cudaMemcpy(&axis, upperNodes->GetInfoData() + i, sizeof(char), cudaMemcpyDeviceToHost);

                    float splitPos;
                    cudaMemcpy(&splitPos, upperNodes->GetSplitPositionData() + i, sizeof(float), cudaMemcpyDeviceToHost);
                        
                    float4 parentAabbMin, parentAabbMax;
                    cudaMemcpy(&parentAabbMin, upperNodes->GetAabbMinData() + i, sizeof(float4), cudaMemcpyDeviceToHost);
                    cudaMemcpy(&parentAabbMax, upperNodes->GetAabbMaxData() + i, sizeof(float4), cudaMemcpyDeviceToHost);
                        
                    int leftIndex;
                    cudaMemcpy(&leftIndex, upperNodes->GetLeftData() + i, sizeof(int), cudaMemcpyDeviceToHost);
                        
                    int2 leftPrimInfo;
                    cudaMemcpy(&leftPrimInfo, upperNodes->GetPrimitiveInfoData() + leftIndex, sizeof(int2), cudaMemcpyDeviceToHost);

                    float4 leftAabbMin = parentAabbMin;
                    float4 leftAabbMax = make_float4(axis == KDNode::X ? splitPos : parentAabbMax.x,
                                                     axis == KDNode::Y ? splitPos : parentAabbMax.y,
                                                     axis == KDNode::Z ? splitPos : parentAabbMax.z,
                                                     parentAabbMax.w);

                    bool leftIsLeaf = leftPrimInfo.y < TriangleLowerNode::MAX_SIZE;
                    for (int j = leftPrimInfo.x; j < leftPrimInfo.x + leftPrimInfo.y; ++j){
                        float4 primMin, primMax;
                        if (leftIsLeaf){
                            cudaMemcpy(&primMin, geom->GetAabbMinData() + j, sizeof(float4), cudaMemcpyDeviceToHost);
                            cudaMemcpy(&primMax, geom->GetAabbMaxData() + j, sizeof(float4), cudaMemcpyDeviceToHost);
                        }else{
                            cudaMemcpy(&primMin, aabbMin->GetDeviceData() + j, sizeof(float4), cudaMemcpyDeviceToHost);
                            cudaMemcpy(&primMax, aabbMax->GetDeviceData() + j, sizeof(float4), cudaMemcpyDeviceToHost);
                        }
                            
                        if (!aabbContains(leftAabbMin, leftAabbMax, primMin))
                            throw Core::Exception("primitive  " + Utils::Convert::ToString(j) + 
                                                  "'s min " + Utils::CUDA::Convert::ToString(primMin) +
                                                  " not included in node " + Utils::Convert::ToString(leftIndex) + 
                                                  "'s aabb " + Utils::CUDA::Convert::ToString(leftAabbMin) +
                                                  " -> " + Utils::CUDA::Convert::ToString(leftAabbMax));

                        if (!aabbContains(leftAabbMin, leftAabbMax, primMax))
                            throw Core::Exception("primitive  " + Utils::Convert::ToString(j) + 
                                                  "'s max " + Utils::CUDA::Convert::ToString(primMax) +
                                                  " not included in left aabb " + Utils::CUDA::Convert::ToString(leftAabbMin)
                                                  + " -> " + Utils::CUDA::Convert::ToString(leftAabbMax));
                    }

                    int rightIndex;
                    cudaMemcpy(&rightIndex, upperNodes->GetRightData() + i, sizeof(int), cudaMemcpyDeviceToHost);
                    CHECK_FOR_CUDA_ERROR();
                        
                    int2 rightPrimInfo;
                    cudaMemcpy(&rightPrimInfo, upperNodes->GetPrimitiveInfoData() + rightIndex, sizeof(int2), cudaMemcpyDeviceToHost);
                    CHECK_FOR_CUDA_ERROR();
                        
                    float4 rightAabbMin = make_float4(axis == KDNode::X ? splitPos : parentAabbMin.x,
                                                      axis == KDNode::Y ? splitPos : parentAabbMin.y,
                                                      axis == KDNode::Z ? splitPos : parentAabbMin.z,
                                                      parentAabbMin.w);
                    float4 rightAabbMax = parentAabbMax;

                    bool rightIsLeaf = rightPrimInfo.y < TriangleLowerNode::MAX_SIZE;
                    for (int j = rightPrimInfo.x; j < rightPrimInfo.x + rightPrimInfo.y; ++j){
                        float4 primMin, primMax;
                        if (rightIsLeaf){
                            cudaMemcpy(&primMin, geom->GetAabbMinData() + j, sizeof(float4), cudaMemcpyDeviceToHost);
                            cudaMemcpy(&primMax, geom->GetAabbMaxData() + j, sizeof(float4), cudaMemcpyDeviceToHost);
                        }else{
                            cudaMemcpy(&primMin, aabbMin->GetDeviceData() + j, sizeof(float4), cudaMemcpyDeviceToHost);
                            cudaMemcpy(&primMax, aabbMax->GetDeviceData() + j, sizeof(float4), cudaMemcpyDeviceToHost);
                        }
                        CHECK_FOR_CUDA_ERROR();
                            
                        if (!aabbContains(rightAabbMin, rightAabbMax, primMin))
                            throw Core::Exception("primitive  " + Utils::Convert::ToString(j) + 
                                                  "'s min " + Utils::CUDA::Convert::ToString(primMin) +
                                                  " not included in right aabb " + Utils::CUDA::Convert::ToString(rightAabbMin)
                                                  + " -> " + Utils::CUDA::Convert::ToString(rightAabbMax));

                        if (!aabbContains(rightAabbMin, rightAabbMax, primMax))
                            throw Core::Exception("primitive  " + Utils::Convert::ToString(j) + 
                                                  "'s max " + Utils::CUDA::Convert::ToString(primMax) +
                                                  " not included in right aabb " + Utils::CUDA::Convert::ToString(rightAabbMin)
                                                  + " -> " + Utils::CUDA::Convert::ToString(rightAabbMax));
                    }
                }
#endif

            }

        }
    }
}
