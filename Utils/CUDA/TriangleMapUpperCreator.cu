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

            TriangleMapUpperCreator::TriangleMapUpperCreator(){
                
                cutCreateTimer(&timerID);

                primMin = new CUDADataBlock<1, float4>(1);
                primMax = new CUDADataBlock<1, float4>(1);
                primIndices = new CUDADataBlock<1, int>(1);

                leafIDs = new CUDADataBlock<1, int>(1);

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
                
                if (leafIDs) delete leafIDs;

                if (aabbMin) delete aabbMin;
                if (aabbMax) delete aabbMax;
                if (tempAabbMin) delete tempAabbMin;
                if (tempAabbMax) delete tempAabbMax;

                if (nodeSegments) delete nodeSegments;

                if (splitSide) delete splitSide;
                if (splitAddr) delete splitAddr;
                if (leafSide) delete leafSide;
                if (leafAddr) delete leafAddr;
                if (childSize) delete childSize;
            }

            namespace KernelsHat {
                #include <Utils/CUDA/Kernels/TriangleUpper.h>
                #include <Utils/CUDA/Kernels/TriangleUpperSegment.h>
                #include <Utils/CUDA/Kernels/ReduceSegments.h>
                #include <Utils/CUDA/Kernels/TriangleUpperChildren.h>
                #include <Utils/CUDA/Kernels/TriangleKernels.h>
            }
            using namespace KernelsHat;

            void TriangleMapUpperCreator::Create(TriangleMap* map, 
                                                 CUDADataBlock<1, int>* upperLeafIDs){

                this->map = map;

                int activeIndex = 0, activeRange = 1;
                int childrenCreated;
                int triangles = map->GetGeometry()->GetSize();

                primMin->Extend(0);
                primMax->Extend(0);

                // Setup root node!
                int2 i = make_int2(0, triangles);
                cudaMemcpy(map->nodes->GetPrimitiveInfoData(), &i, sizeof(int2), cudaMemcpyHostToDevice);
                int parent = 0;
                cudaMemcpy(map->nodes->GetParentData(), &parent, sizeof(int), cudaMemcpyHostToDevice);
                map->nodes->Resize(1);

                // Setup bounding box info
                aabbMin->Extend(triangles);
                aabbMax->Extend(triangles);

                // @TODO calc max and min here
                
                unsigned int blocks, threads;
                Calc1DKernelDimensions(triangles, blocks, threads);
                AddIndexToAabb<<<blocks, threads>>>(map->GetGeometry()->GetAabbMinData(), triangles, aabbMin->GetDeviceData());
                cudaMemcpy(aabbMax->GetDeviceData(), map->GetGeometry()->GetAabbMaxData(), 
                           triangles * sizeof(float4), cudaMemcpyDeviceToDevice);
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
                int triangles = primMin->GetSize();
                //logger.info << "=== Process " << activeRange << " Upper Nodes Starting at " << activeIndex << " === with " << triangles << " primitives" << logger.end;

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
                segments.size = amountOfSegments;

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
                
        }
    }
}
