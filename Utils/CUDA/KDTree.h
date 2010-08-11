// KD tree structs for CUDA
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// Modified by Anders Bach Nielsen <abachn@daimi.au.dk> - 21. Nov 2007
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _CUDA_KD_TREE_H_
#define _CUDA_KD_TREE_H_

#include <Meta/CUDA.h>

struct KDTriNode {
    char axis;
    float pos; // position along that axis
    //void* left, right; // children
};

struct KDTriLeaf {
    float3 vert[3];
};

struct photon {
    float3 pos;
};

#endif // _CUDA_KD_TREE_H_
