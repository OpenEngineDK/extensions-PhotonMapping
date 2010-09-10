// KD tree upper node for photons
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Scene/KDNode.h>

namespace OpenEngine {
    namespace Scene {
        
        KDNode::KDNode()
            : maxSize(0), size(0) {}

        KDNode::KDNode(int i)
            : maxSize(i), size(0) {
            
            cudaMalloc(&info, maxSize * sizeof(char));
            cudaMalloc(&splitPos, maxSize * sizeof(float));
            cudaMalloc(&aabbMin, maxSize * sizeof(point));
            cudaMalloc(&aabbMax, maxSize * sizeof(point));

            cudaMalloc(&photonIndex, maxSize * sizeof(int));

            CHECK_FOR_CUDA_ERROR();
        }

        void KDNode::Resize(int i){
            int copySize = min(i, size);
            
            char *tempChar;
            float *tempFloat;
            point *tempPoint;
            int *tempInt;

            cudaMalloc(&tempChar, i * sizeof(char));
            cudaMemcpy(tempChar, info, copySize * sizeof(char), cudaMemcpyDeviceToDevice);
            cudaFree(info);
            info = tempChar;
            CHECK_FOR_CUDA_ERROR();

            cudaMalloc(&tempFloat, i * sizeof(float));
            cudaMemcpy(tempFloat, splitPos, copySize * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaFree(splitPos);
            splitPos = tempFloat;
            CHECK_FOR_CUDA_ERROR();

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
            cudaMemcpy(tempInt, photonIndex, copySize * sizeof(int), cudaMemcpyDeviceToDevice);
            cudaFree(photonIndex);
            photonIndex = tempInt;
            CHECK_FOR_CUDA_ERROR();

            maxSize = i;
            size = copySize;
        }
        
    }
}
