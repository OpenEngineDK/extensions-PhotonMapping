// KD tree structs for CUDA
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// Modified by Anders Bach Nielsen <abachn@daimi.au.dk> - 21. Nov 2007
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _CUDA_PHOTON_UTILS_H_
#define _CUDA_PHOTON_UTILS_H_

#include <algorithm>

#define fInfinity 0x7f800000

unsigned int NextPow2(unsigned int x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

void Calc1DKernelDimensions(const unsigned int size, 
                            unsigned int &blocks, unsigned int &threads){
    unsigned int MAX_THREADS = activeCudaDevice.maxThreadsDim[0];
    unsigned int MAX_BLOCKS = (activeCudaDevice.maxGridSize[0]+1) / MAX_THREADS;

    threads = (size < MAX_THREADS * 2) ? NextPow2((size + 1)/ 2) : MAX_THREADS;
    blocks = (size + (threads * 2 - 1)) / (threads * 2);
    blocks = min(MAX_BLOCKS, blocks);
}


__host__ __device__ float3 max(float3 v, float3 u){
    return make_float3(max(v.x, u.x),
                       max(v.y, u.y),
                       max(v.z, u.z));
}

__host__ __device__ float3 min(float3 v, float3 u){
    return make_float3(min(v.x, u.x),
                       min(v.y, u.y),
                       min(v.z, u.z));
}


#endif // _CUDA_PHOTON_UTILS_H_
