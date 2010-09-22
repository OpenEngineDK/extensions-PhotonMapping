// KD tree upper node leaf list for photons
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Utils/CUDA/UpperNodeLeafList.h>

#include <Utils/CUDA/Utils.h>

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {

            UpperNodeLeafList::UpperNodeLeafList(int size)
                : maxSize(size), size(0) {
                cudaMalloc(&leafIDs, size * sizeof(int));
            }
            
            void UpperNodeLeafList::Resize(int s){
                int copySize = min(size, s);
                int *tempInt;
                cudaSafeMalloc(&tempInt, s * sizeof(int));
                cudaMemcpy(tempInt, leafIDs, copySize * sizeof(int), cudaMemcpyDeviceToDevice);
                cudaFree(leafIDs);
                leafIDs = tempInt;
                CHECK_FOR_CUDA_ERROR();

                maxSize = s;
                size = copySize;
            }

        }
    }
}
