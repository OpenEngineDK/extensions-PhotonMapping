// KD tree upper node for photons
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _CUDA_KD_PHOTON_UPPER_NODE_H_
#define _CUDA_KD_PHOTON_UPPER_NODE_H_

#include <Meta/CUDA.h>
#include <string>
#include <Utils/CUDA/Convert.h>
#include <Utils/CUDA/KDTree.h>

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {
            namespace Photon {

                class KDPhotonUpperNode {
                public:
                    static const unsigned int LEAF = 0;
                    static const unsigned int X = 1;
                    static const unsigned int Y = 2;
                    static const unsigned int Z = 3;

                    char *info; // 0 = LEAF,1 = X, 2 = Y, 3 = Z. 6 bits left for stuff
                    float *splitPos; // position along that axis
                    float3 *aabbMin; // Min cornor of aabb, important?
                    float3 *aabbMax; // Max cornor of aabb, important?
                    unsigned int *startIndex; // index into photons that this node starts at.
                    unsigned int *range; // Range of photons that the node spans
                    unsigned int *parent;
                    unsigned int *child; // If it is a node then child points to the
                    // left child, if it is a 'leaf' then the child
                    // is the lower node.
                    //unsigned int *left, *right; //children, will right not always be left+1? If so remove the bloody variable
                    //unsigned int *lower; // index into the lower node, use max or 0 as 'not used'?
                    unsigned int maxSize; // maximum number of nodes. Should be increased on demand.
                    unsigned int size;

                public:
                    void Init(unsigned int size);
                    
                    std::string ToString(unsigned int i);

                    void CheckBoundingBox(unsigned int i, photon photons);
                    
                };

            }
        }
    }
}
#endif _CUDA_KD_PHOTON_UPPER_NODE_H_