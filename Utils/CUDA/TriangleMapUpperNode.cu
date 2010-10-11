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
#include <Utils/CUDA/Kernels/TriangleUpperSegment.h>

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {

            using namespace Kernels;

            void TriangleMap::CreateUpperNodes(){
                int activeIndex = 0, activeRange = 1;
                int newActiveIndex, childrenCreated;

                // Setup root node!
                int2 i = make_int2(0, triangles);
                cudaMemcpy(upperNodes->GetPrimitiveInfoData(), &i, sizeof(int2), cudaMemcpyHostToDevice);
                upperNodes->size = 1;

                while (activeRange > 0){
                    ProcessUpperNodes(activeIndex, activeRange, 
                                      newActiveIndex, childrenCreated);

                    // @TODO Isn't active index = upperNodes.size - childrenCreated?
                    activeIndex = newActiveIndex;
                    activeRange = childrenCreated;
                }

            }

            void TriangleMap::ProcessUpperNodes(int activeIndex, int activeRange, 
                                                int &leafsCreated, int &childrenCreated){

                logger.info << "=== Process " << activeRange << " Upper Nodes Starting at " << activeIndex << " === with " << triangles << " triangles" << logger.end;

                // Copy bookkeeping to symbols
                cudaMemcpyToSymbol(d_activeNodeIndex, &activeIndex, sizeof(int));
                cudaMemcpyToSymbol(d_activeNodeRange, &activeRange, sizeof(int));
                CHECK_FOR_CUDA_ERROR();

                Segment(activeIndex, activeRange);
                CHECK_FOR_CUDA_ERROR();

                // Calculate aabb

                // Calculate children placement
                
                leafsCreated = 0;
                childrenCreated = 0;
            }

            void TriangleMap::Segment(int activeIndex, int activeRange){
                if (nodeSegments->GetSize() < (unsigned int)activeRange+1)
                    nodeSegments->Resize(activeRange+1);

                //logger.info <<" primitive info " << Utils::CUDA::Convert::ToString(upperNodes->GetPrimitiveInfoData() + activeIndex, activeRange) << logger.end;

                unsigned int blocks, threads;
                Calc1DKernelDimensions(activeRange, blocks, threads);
                NodeSegments<<<blocks, threads>>>(upperNodes->GetPrimitiveInfoData() + activeIndex,
                                                  nodeSegments->GetDeviceData());

                //logger.info <<" node segments " << Utils::CUDA::Convert::ToString(nodeSegments->GetDeviceData(), activeRange) << logger.end;

                CHECK_FOR_CUDA_ERROR();
                cudppScan(scanHandle, nodeSegments->GetDeviceData(), nodeSegments->GetDeviceData(), activeRange+1);
                
                //logger.info <<" node segments addrs " << Utils::CUDA::Convert::ToString(nodeSegments->GetDeviceData(), activeRange+1) << logger.end;

                int amountOfSegments;
                cudaMemcpy(&amountOfSegments, nodeSegments->GetDeviceData() + activeRange, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpyToSymbol(d_segments, nodeSegments->GetDeviceData() + activeRange, sizeof(int), 0, cudaMemcpyDeviceToDevice);
                CHECK_FOR_CUDA_ERROR();

                //logger.info << " Number of segments: " << amountOfSegments << logger.end;

                if (segments.size < amountOfSegments)
                    segments.Resize(amountOfSegments);

                cudaMemset(segments.GetOwnerData(), 0, amountOfSegments * sizeof(int));
                MarkOwnerStart<<<blocks, threads>>>(segments.GetOwnerData(),
                                                    nodeSegments->GetDeviceData());
                CHECK_FOR_CUDA_ERROR();

                cudppScan(scanInclHandle, segments.GetOwnerData(), segments.GetOwnerData(), amountOfSegments);

                //logger.info << " Segment owners " << Utils::CUDA::Convert::ToString(segments.GetOwnerData(), amountOfSegments) << logger.end;

                Calc1DKernelDimensions(amountOfSegments, blocks, threads);
                CalcSegmentPrimitives<<<blocks, threads>>>(segments.GetOwnerData(),
                                                           nodeSegments->GetDeviceData(),
                                                           upperNodes->GetPrimitiveInfoData() + activeIndex,
                                                           segments.GetPrimitiveInfoData());
                CHECK_FOR_CUDA_ERROR();

                //logger.info << " Segment primitive info " << Utils::CUDA::Convert::ToString(segments.GetPrimitiveInfoData(), amountOfSegments) << logger.end;

            }

        }
    }
}
