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

#include <Utils/CUDA/Kernels/PhotonMapDeviceVars.h>
#include <Utils/CUDA/Kernels/TriangleUpperSegment.h>

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {

            using namespace Kernels;

            void TriangleMap::ProcessUpperNodes(int activeIndex, int activeRange, 
                                                int &leafsCreated, int &childrenCreated){
                leafsCreated = 0;
                childrenCreated = 0;
            }

            void TriangleMap::Segment(int activeIndex, int activeRange){
                if (nodeSegments->GetSize() < activeRange)
                    nodeSegments->Resize(activeRange);

                unsigned int blocks, threads;
                Calc1DKernelDimensions(activeRange, blocks, threads);
                NodeSegments<<<blocks, threads>>>(upperNodes->GetPrimitiveInfoData() + activeIndex,
                                                  nodeSegments->GetDeviceData());
                CHECK_FOR_CUDA_ERROR();
                cudppScan(scanHandle, nodeSegments->GetDeviceData(),  nodeSegments->GetDeviceData(), activeRange+1);
                
                
            }

        }
    }
}
