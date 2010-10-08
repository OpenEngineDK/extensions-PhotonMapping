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

using namespace OpenEngine::Resources::CUDA;

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {

            class Segments {
            public:
                
                static const int SEGMENT_SIZE = 256;
                
                CUDADataBlock<1, int> *nodeIDs;
                CUDADataBlock<1, int2> *photonInfo;
                // Variables for holding the intermediate min/max values
                CUDADataBlock<1, point> *aabbMin, *aabbMax;
                CUDADataBlock<1, int> *prefixSum; // prefix sum.

                int maxSize, size;

            public:
                Segments();
                Segments(int i);

                void Resize(int i);
                
            };

        }
    }
}

#endif
