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
#include <Utils/CUDA/NodeChildren.h>
#include <Utils/CUDA/UpperNodeLeafList.h>

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
                int *photonOwners;
                int *newOwners;
                PhotonUpperNode upperNodes;
                PhotonLowerNode lowerNodes;
                
                NodeChildren tempChildren;
                UpperNodeLeafList upperNodeLeafList;

                CUDPPConfiguration scanConfig;
                CUDPPHandle scanHandle;
                CUDPPConfiguration sortConfig;
                CUDPPHandle sortHandle;

                // Sorted photon positions
                float *xIndices, *yIndices, *zIndices;
                float *xKeys, *yKeys, *zKeys;
                float4 *xSorted, *ySorted, *zSorted;

                int *leafSide;
                int *leafPrefix;
                int *splitSide;
                int *prefixSum;
                int *splitLeft;
                int2 *splitAddrs;
                float4* tempPhotonPos;

            public:
                PhotonMap(unsigned int size);

                void Create();

            private:
                void SortPhotons();

                void ProcessUpperNodes(int activeIndex,
                                       int activeRange,
                                       int unhandledLeafs,
                                       int &leafsCreated,
                                       int &childrenCreated,
                                       int &activePhotons);

                void ComputeBoundingBox(int activeIndex,
                                        int activeRange);

                void SplitUpperNodePhotons(int activeIndex,
                                           int activeRange,
                                           int unhandledLeafs,
                                           int &activePhotons);

                void SplitSortedArray(float4 *&sortedArray, int activePhotons);

                void SetupUpperLeafNodes(int activeIndex,
                                         int leafNodes,
                                         int photonOffset);

                void CreateChildren(int activeIndex,
                                    int activeRange,
                                    int activePhotons,
                                    int &leafsCreated,
                                    int &childrenCreated);

                void PreprocessLowerNodes();

                void ProcessLowerNodes(int activeIndex,
                                       int activeRange,
                                       int &childrenCreated);
                                

                void VerifyMap();
                int VerifyUpperNode(int i, char info, float splitPos,
                                    point parentAABBMin, point parentAABBMax);
            };

        }
    }
}

#endif
