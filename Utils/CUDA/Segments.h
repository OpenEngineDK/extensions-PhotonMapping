// Variables used when segmenting the upper nodes
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _CUDA_NODE_SEGMENTS_H_
#define _CUDA_NODE_SEGMENTS_H_

#include <Utils/CUDA/Point.h>
#include <Resources/CUDA/CUDADataBlock.h>

#include <string>

using namespace OpenEngine::Resources::CUDA;

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {

            class Segments {
            public:                

#define _D_SEGMENT_SIZE 256
                static const int SEGMENT_SIZE = _D_SEGMENT_SIZE;

                CUDADataBlock<int> *nodeIDs;
                CUDADataBlock<int2> *primitiveInfo;
                // Variables for holding the intermediate min/max values
                CUDADataBlock<float4> *aabbMin, *aabbMax;
                CUDADataBlock<int> *prefixSum; // prefix sum.

                int maxSize, size;

            public:
                Segments();
                Segments(int i);

                inline int GetSize() const { return size; }

                void IncreaseNodeIDs(const int step);

                void Resize(const int i);
                inline void Extend(const int i) {
                    if (maxSize < i) Resize(i); 
                    else size = i ;
                }

                inline int* GetOwnerData() const { return nodeIDs->GetDeviceData(); }
                inline int2* GetPrimitiveInfoData() const {return primitiveInfo->GetDeviceData(); }
                inline float4* GetAabbMinData() const { return aabbMin->GetDeviceData(); }
                inline float4* GetAabbMaxData() const { return aabbMax->GetDeviceData(); }
                
                std::string ToString(int i);

            };

        }
    }
}

#endif
