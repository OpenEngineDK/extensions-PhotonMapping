// KD tree lower node for photons
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Scene/PhotonLowerNode.h>
#include <Utils/CUDA/Utils.h>

namespace OpenEngine {
    namespace Scene {
     
        PhotonLowerNode::PhotonLowerNode()
            : KDNode() {}

        PhotonLowerNode::PhotonLowerNode(int photons)
            : KDNode(0) {

            maxSize = photons / MAX_SIZE * (2 * MAX_SIZE - 1);
            KDNode::Resize(maxSize);
            
            cudaSafeMalloc(&smallRoot, this->maxSize * sizeof(int));

            // Alloc split information
            cudaSafeMalloc(&splitTriangleSetX, photons * sizeof(int2));
            cudaSafeMalloc(&splitTriangleSetY, photons * sizeof(int2));
            cudaSafeMalloc(&splitTriangleSetZ, photons * sizeof(int2));

            CHECK_FOR_CUDA_ERROR();
        }

        void PhotonLowerNode::Resize(int i){
            KDNode::Resize(i);

            unsigned int copySize = this->size;

            int *tempInt;
            cudaSafeMalloc(&tempInt, i * sizeof(int));
            cudaMemcpy(tempInt, smallRoot, copySize * sizeof(int), cudaMemcpyDeviceToDevice);
            cudaFree(smallRoot);
            smallRoot = tempInt;
            CHECK_FOR_CUDA_ERROR();
        }

    }
}
