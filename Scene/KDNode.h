// KD tree upper node for photons
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _CUDA_KD_BASE_NODE_H_
#define _CUDA_KD_BASE_NODE_H_

#include <Meta/CUDA.h>
#include <Utils/Cuda/types.h>

namespace OpenEngine {
    namespace Scene {
        
        class KDNode {
        public:
            static const unsigned int LEAF = 0;
            static const unsigned int X = 1;
            static const unsigned int Y = 2;
            static const unsigned int Z = 3;

            char *info; // 0 = LEAF,1 = X, 2 = Y, 3 = Z. 6 bits left for stuff
            float *splitPos; // position along that axis
            point *aabbMin, *aabbMax;
            int *photonIndex;

            int maxSize;
            int size;

        public:
            KDNode();
            KDNode(int i);

            virtual void Resize(int i);

        };
        
    }
}

#endif
