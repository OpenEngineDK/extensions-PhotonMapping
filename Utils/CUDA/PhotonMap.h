// Photon map class for CUDA
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _PHOTON_MAP_H_
#define _PHoTON_MAP_H_

#include <Meta/CUDA.h>
#include <Scene/PhotonNode.h>
#include <Scene/PhotonUpperNode.h>
#include <Scene/PhotonLowerNode.h>
#include <Utils/CUDA/AABBVar.h>
#include <Utils/CUDA/SplitVar.h>

#include <Meta/CUDPP.h>

using namespace OpenEngine::Scene;

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {

            class PhotonMap {
            public:

                // Constants
                unsigned int MAX_BLOCKS;

                unsigned int timerID;

                PhotonNode photons;
                PhotonUpperNode upperNodes;
                PhotonLowerNode lowerNodes;
                AABBVar aabbVars;

                CUDPPConfiguration scanConfig;
                CUDPPHandle scanHandle;

                // The approximate range
                int approxPhotonRange;

            public:
                PhotonMap(unsigned int size);

                void Create();

            private:
                void ProcessUpperNodes(unsigned int activeIndex,
                                       unsigned int activeRange,
                                       unsigned int &childrenCreated);

                void ComputeBoundingBox(unsigned int activeIndex,
                                        unsigned int activeRange);

                void SplitUpperNodePhotons(unsigned int activeIndex,
                                           unsigned int activeRange);

                unsigned int CreateChildren(unsigned int activeIndex,
                                            unsigned int activeRange);

                void PreprocessLowerNodes(unsigned int range);

                void ProcessLowerNodes(unsigned int activeIndex,
                                       unsigned int activeRange,
                                       unsigned int &childrenCreated);
                                
            }

        }
    }
}

#endif
