// Variables used when segmenting the upper nodes
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Utils/CUDA/Segments.h>

#include <Meta/CUDA.h>

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {
            
            Segments::Segments()
                : maxSize(0){}

            Segments::Segments(int i)
                : maxSize(i){
                
                nodeIDs = new CUDADataBlock<1, int>(i);
                photonInfo = new CUDADataBlock<1, int2>(i);
                aabbMin = new CUDADataBlock<1, point>(i);
                aabbMax = new CUDADataBlock<1, point>(i);
                prefixSum = new CUDADataBlock<1, int>(i);
            }

            void Segments::Resize(int i){
                nodeIDs->Resize(i);
                photonInfo->Resize(i);
                aabbMin->Resize(i);
                aabbMax->Resize(i);
                prefixSum->Resize(i);
                
                maxSize = i;
            }

        }
    }
}
