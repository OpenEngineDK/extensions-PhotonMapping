// List of splitting planes.
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Utils/CUDA/SplittingPlanes.h>

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {

            SplittingPlanes::SplittingPlanes()
                : size(0) {}

            SplittingPlanes::SplittingPlanes(unsigned int i)
                : size(i) {
                cudaMalloc(&triangleSetX, size * sizeof(int2));
                cudaMalloc(&triangleSetY, size * sizeof(int2));
                cudaMalloc(&triangleSetZ, size * sizeof(int2));
            }

            void SplittingPlanes::Resize(unsigned int i){
                unsigned int copySize = min(i, size);

                int2 *temp;
                cudaMalloc(&temp, i * sizeof(int2));
                cudaMemcpy(temp, triangleSetX, copySize * sizeof(int2), cudaMemcpyDeviceToDevice);
                cudaFree(triangleSetX);
                triangleSetX = temp;
                CHECK_FOR_CUDA_ERROR();

                cudaMalloc(&temp, i * sizeof(int2));
                cudaMemcpy(temp, triangleSetY, copySize * sizeof(int2), cudaMemcpyDeviceToDevice);
                cudaFree(triangleSetY);
                triangleSetY = temp;
                CHECK_FOR_CUDA_ERROR();

                cudaMalloc(&temp, i * sizeof(int2));
                cudaMemcpy(temp, triangleSetZ, copySize * sizeof(int2), cudaMemcpyDeviceToDevice);
                cudaFree(triangleSetZ);
                triangleSetZ = temp;
                CHECK_FOR_CUDA_ERROR();
            }
            
        }
    }
}
