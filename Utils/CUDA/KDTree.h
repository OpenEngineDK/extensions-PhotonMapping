// KD tree structs for CUDA
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _CUDA_KD_TREE_HCU_
#define _CUDA_KD_TREE_HCU_

#include <Meta/CUDA.h>
#include <string>
#include <sstream>
#include <Utils/CUDA/Convert.h>

namespace OpenEngine {
    namespace Resources {
        class IDataBlock;
    }
}


// Constants
const unsigned int MAX_THREADS = 256;
const unsigned int MAX_BLOCKS = 64;
const unsigned int BUCKET_SIZE = 32; // size of buckets in lower nodes

struct KDPhotonNodeLower {
    char *axis;
    float *pos; // position along that axis
    unsigned int *left, *right; // children, right necessary?
};

/**
 * Leaf and axis (plus even child) can be placed in one unsigned
 * int. Will probably be a slowdown but use less space.
 *
 * Place is-left or is-right node info in 'info' as a bit?
 */
struct photon {
    // Association list between indices used in the tree and actual
    // photons. This makes it impossible to coalesce memory
    // read/writes (for photons, some performance may be gained by
    // coalescing assoc lookups)
    //unsigned int *assoc;
    float3* pos;
    unsigned int maxSize;
    unsigned int size;
};


void InitPhotons(unsigned int amount);

void MapPhotons();

void MapPhotonsToOpenGL(OpenEngine::Resources::IDataBlock* pos);

#endif // _CUDA_KD_TREE_HCU_
