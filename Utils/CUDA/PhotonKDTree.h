// Photon KD tree class for CUDA
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _PHOTON_KD_TREE_H_
#define _PHoTON_KD_TREE_H_

#include <Meta/CUDA.h>
#include <Scene/PhotonNode.h>
#include <Scene/KDPhotonUpperNode.h>
#include <Utils/CUDA/AABBVar.h>
#include <Utils/CUDA/SplitVar.h>

using namespace OpenEngine::Scene;

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {

            class PhotonKDTree {
            public:

                // Constants
                static const unsigned int MAX_THREADS = 256;
                static const unsigned int MAX_BLOCKS = 64;
                static const unsigned int BUCKET_SIZE = 32; // size of buckets in lower nodes

                PhotonNode photons;
                KDPhotonUpperNode upperNodes;
                //PhotonLowerKDNode lowerNodes;
                AABBVar aabbVars;
                SplitVar splitVars;
                
            public:
                PhotonKDTree(unsigned int size);

                void Create();

            private:
                void CreateUpperNodes(unsigned int activeIndex, 
                                      unsigned int activeRange, 
                                      unsigned int &childrenCreated, 
                                      unsigned int &lowerCreated);

                void CreateLowerNodes();

                void ComputeBoundingBoxes(unsigned int activeIndex,
                                          unsigned int activeRange);

                void SplitUpperNodes(unsigned int activeIndex,
                                     unsigned int activeRange);
            };
            
        }
    }
}

#endif
