// KD tree lower node for photons
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _CUDA_KD_PHOTON_LOWER_NODE_H_
#define _CUDA_KD_PHOTON_LOWER_NODE_H_

#include <Scene/KDNode.h>

namespace OpenEngine {
    namespace Scene {
        
        class PhotonLowerNode : public KDNode {
        public:
            static const int SEARCH_RADIUS = 50;
            //static const int MAX_SIZE = 32;
            static const int MAX_SIZE = 32;

            //unsigned int *photonBitmap;
            int *smallRoot;
            //int *left, *right;

        public:
            PhotonLowerNode();
            PhotonLowerNode(int i);

            void Resize(int i);
        };

    }
}

#endif
