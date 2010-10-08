// KD tree upper node for photons
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Scene/KDNode.h>
#include <Utils/CUDA/Utils.h>
#include <Logging/Logger.h>

namespace OpenEngine {
    namespace Scene {
        
        KDNode::KDNode()
            : maxSize(0), size(0) {}

        KDNode::KDNode(int i)
            : maxSize(i), size(0) {
            
            info = new CUDADataBlock<1, char>(maxSize);
            splitPos = new CUDADataBlock<1, float>(maxSize);
            aabbMin = new CUDADataBlock<1, point>(maxSize);
            aabbMax = new CUDADataBlock<1, point>(maxSize);
            photonInfo = new CUDADataBlock<1, int2>(maxSize);

            left = new CUDADataBlock<1, int>(maxSize);
            right = new CUDADataBlock<1, int>(maxSize);

            CHECK_FOR_CUDA_ERROR();
        }

        KDNode::~KDNode(){
            logger.info << "Oh noes the KDNode is gone man" << logger.end;
            delete info;
            delete splitPos;
            delete aabbMin;
            delete aabbMax;
            delete photonInfo;
            delete left;
            delete right;
        }

        void KDNode::Resize(int i){
            int copySize = min(i, size);
            
            info->Resize(i);
            splitPos->Resize(i);
            aabbMin->Resize(i);
            aabbMax->Resize(i);
            photonInfo->Resize(i);
            left->Resize(i);
            right->Resize(i);

            maxSize = i;
            size = copySize;
        }
        
    }
}
