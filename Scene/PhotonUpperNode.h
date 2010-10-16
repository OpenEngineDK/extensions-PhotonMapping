// KD tree upper node for photons
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _CUDA_PHOTON_UPPER_NODE_H_
#define _CUDA_PHOTON_UPPER_NODE_H_

#include <Meta/CUDA.h>
#include <Scene/KDNode.h>
#include <Utils/CUDA/Convert.h>
#include <Scene/PhotonNode.h>

namespace OpenEngine {
    namespace Resources {
        class IDataBlock;
    }
    namespace Scene {

        class PhotonUpperNode : public KDNode {
        public:
            //int *parents;//, *tempParent;

        public:
            PhotonUpperNode();
            PhotonUpperNode(int size);

            void Resize(int i);

            void MapToDataBlocks(Resources::IDataBlock* vertices,
                                 Resources::IDataBlock* colors);

            std::string PhotonsToString(unsigned int i, PhotonNode photons);
        };

    }
}
#endif // _CUDA_KD_PHOTON_UPPER_NODE_H_
