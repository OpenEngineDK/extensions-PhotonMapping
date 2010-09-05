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

#include <Meta/CUDPP.h>

using namespace OpenEngine::Scene;

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {

            class PhotonKDTree {
            public:

                // Constants
                unsigned int MAX_BLOCKS;

                unsigned int timerID;

                PhotonNode photons;
                KDPhotonUpperNode upperNodes;
                //PhotonLowerKDNode lowerNodes;
                AABBVar aabbVars;
                SplitVar splitVars;
                CUDPPConfiguration scanConfig;
                CUDPPHandle scanHandle;
                
            public:
                PhotonKDTree(unsigned int size);

                void Create();

            private:
                void CreateUpperNodes(unsigned int activeIndex, 
                                      unsigned int activeRange, 
                                      unsigned int &childrenCreated, 
                                      unsigned int &lowerCreated);

                void ComputeBoundingBoxes(unsigned int activeIndex,
                                          unsigned int activeRange,
                                          unsigned int *photonRanges);

                void SegmentedReduce(unsigned int blocksUsed);

                unsigned int CreateChildren(unsigned int activeIndex,
                                            unsigned int activeRange);

                void SplitUpperNodePhotons(unsigned int activeIndex,
                                           unsigned int activeRange,
                                           unsigned int *photonRanges);

                void CreateLowerNodes();

            };
            
        }
    }
}

#endif
