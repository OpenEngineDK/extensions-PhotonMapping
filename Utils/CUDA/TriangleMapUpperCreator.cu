// Triangle map upper node creator interface for CUDA
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Utils/CUDA/TriangleMapUpperCreator.h>

namespace OpenEngine {    
    namespace Utils {
        namespace CUDA {

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

            void TriangleMapUpperCreator::Create(TriangleMap* map, 
                                                 CUDADataBlock<1, int>* upperLeafIDs){

                this->map = map;

                int activeIndex = 0, activeRange = 1;
                int childrenCreated;

            }

        }
    }
}
