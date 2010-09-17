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
                : maxSize(0), size(0) {}

            Segments::Segments(int i)
                : maxSize(i), size(i){
                cudaMalloc(&nodeIDs, maxSize * sizeof(int));
                cudaMalloc(&photonIndices, maxSize * sizeof(int));
                cudaMalloc(&photonRanges, maxSize * sizeof(int));

                cudaMalloc(&aabbMin, maxSize * sizeof(point));
                cudaMalloc(&aabbMax, maxSize * sizeof(point));
                
                cudaMalloc(&prefixSum, maxSize * sizeof(int));
            }

            void Segments::Resize(int i){
                int copySize = min(i, size);
                
                int* tempInt;
                cudaMalloc(&tempInt, i * sizeof(int));
                cudaMemcpy(tempInt, nodeIDs, copySize * sizeof(int), cudaMemcpyDeviceToDevice);
                cudaFree(nodeIDs);
                nodeIDs = tempInt;
                CHECK_FOR_CUDA_ERROR();

                cudaMalloc(&tempInt, i * sizeof(int));
                cudaMemcpy(tempInt, photonIndices, copySize * sizeof(int), cudaMemcpyDeviceToDevice);
                cudaFree(photonIndices);
                photonIndices = tempInt;
                CHECK_FOR_CUDA_ERROR();

                cudaMalloc(&tempInt, i * sizeof(int));
                cudaMemcpy(tempInt, photonRanges, copySize * sizeof(int), cudaMemcpyDeviceToDevice);
                cudaFree(photonRanges);
                photonRanges = tempInt;
                CHECK_FOR_CUDA_ERROR();

                point *tempPoint;
                cudaMalloc(&tempPoint, i * sizeof(point));
                cudaMemcpy(tempPoint, aabbMin, copySize * sizeof(point), cudaMemcpyDeviceToDevice);
                cudaFree(aabbMin);
                aabbMin = tempPoint;
                CHECK_FOR_CUDA_ERROR();

                cudaMalloc(&tempPoint, i * sizeof(point));
                cudaMemcpy(tempPoint, aabbMax, copySize * sizeof(point), cudaMemcpyDeviceToDevice);
                cudaFree(aabbMax);
                aabbMax = tempPoint;
                CHECK_FOR_CUDA_ERROR();

                cudaMalloc(&tempInt, i * sizeof(int));
                cudaMemcpy(tempInt, prefixSum, copySize * sizeof(int), cudaMemcpyDeviceToDevice);
                cudaFree(prefixSum);
                prefixSum = tempInt;
                CHECK_FOR_CUDA_ERROR();
                
                maxSize = i;
                size = copySize;
            }

        }
    }
}
