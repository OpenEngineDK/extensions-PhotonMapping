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
#include <string>
#include <Utils/CUDA/Convert.h>
#include <Scene/PhotonNode.h>

namespace OpenEngine {
    namespace Resources {
        class IDataBlock;
    }
    namespace Scene {

        class PhotonUpperNode : public KDNode {
        public:
            static const unsigned int BUCKET_SIZE = 32; // size of buckets in lower nodes
            //static const unsigned int BUCKET_SIZE = 2; // size of buckets in lower nodes
            
            int *range;//, *tempRange; // Range of photons that the node spans
            int *parent;//, *tempParent;
            int *left, *right; // if it is a leaf node then both nodes
                               // point to it's lower node.

        public:
            PhotonUpperNode();
            PhotonUpperNode(int size);

            void Resize(int i);

            /*                    
            void MapToDataBlocks(Resources::IDataBlock* vertices,
                                 Resources::IDataBlock* colors);

            std::string ToString(unsigned int i);
            std::string PhotonsToString(unsigned int i, PhotonNode photons);

            void CheckBoundingBox(unsigned int i, PhotonNode photons);
            */
        };

    }
}
#endif // _CUDA_KD_PHOTON_UPPER_NODE_H_
