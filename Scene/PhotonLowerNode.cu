// KD tree lower node for photons
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Scene/PhotonLowerNode.h>

namespace OpenEngine {
    namespace Scene {
     
        PhotonLowerNode::PhotonLowerNode()
            : KDNode() {}

        PhotonLowerNode::PhotonLowerNode(int i)
            : KDNode(0) {
            
            cudaMalloc(&photonBitmap, this->maxSize * sizeof(unsigned int));
            cudaMalloc(&smallRoot, this->maxSize * sizeof(int));

            CHECK_FOR_CUDA_ERROR();
        }

        void PhotonLowerNode::Resize(int i){
            KDNode::Resize(i);

            unsigned int copySize = this->size;

            unsigned int *tempUint;
            cudaMalloc(&tempUint, i * sizeof(unsigned int));
            cudaMemcpy(tempUint, photonBitmap, copySize * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
            cudaFree(photonBitmap);
            photonBitmap = tempUint;
            CHECK_FOR_CUDA_ERROR();

            int *tempInt;
            cudaMalloc(&tempInt, i * sizeof(int));
            cudaMemcpy(tempInt, smallRoot, copySize * sizeof(int), cudaMemcpyDeviceToDevice);
            cudaFree(smallRoot);
            smallRoot = tempInt;
            CHECK_FOR_CUDA_ERROR();
        }

    }
}
