// Utils for CUDA
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _CUDA_PHOTON_UTILS_H_
#define _CUDA_PHOTON_UTILS_H_

#include <Meta/CUDA.h>

#include <algorithm>

#define fInfinity 0x7f800000

#ifdef OE_SAFE
#define cudaSafeMalloc(ptr, size); cudaMalloc(ptr, size); cudaMemset(*ptr, 127, size);
#else
#define cudaSafeMalloc(ptr, size); cudaMalloc(ptr, size);
#endif

// easy timer defs
#define START_TIMER(timerID) cutResetTimer(timerID);cutStartTimer(timerID);
#define PRINT_TIMER(timerID, name)              \
    cudaThreadSynchronize();                    \
    cutStopTimer(timerID);                      \
    logger.info << name << " time: " << cutGetTimerValue(timerID) << "ms" << logger.end; \
    

inline unsigned int NextPow2(unsigned int x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

inline void Calc1DKernelDimensions(const unsigned int size, 
                                   unsigned int &blocks, unsigned int &threads,
                                   unsigned int maxThreads = 0){
    const unsigned int MAX_THREADS = maxThreads ? maxThreads : activeCudaDevice.maxThreadsDim[0];
    const unsigned int MAX_BLOCKS = activeCudaDevice.maxGridSize[0];

    unsigned int s = NextPow2(size);
    threads = (s < MAX_THREADS) ? s : MAX_THREADS;
    s /= threads;
    blocks = s < MAX_BLOCKS ? s : MAX_BLOCKS;
}

/**
 * Stolen from
 * http://gurmeetsingh.wordpress.com/2008/08/05/fast-bit-counting-routines
 *
 * How and why it works? Magic! Now go code something.
 */
inline __host__ __device__ int bitcount(unsigned int n){
    unsigned int tmp = n - ((n >> 1) & 033333333333)
        - ((n >> 2) & 011111111111);
    return ((tmp + (tmp >> 3)) & 030707070707) % 63;
}

inline __host__ __device__ void maxCorner(float4 v, float3 u, float3 &ret){
    ret = make_float3(max(v.x, u.x),
                      max(v.y, u.y),
                      max(v.z, u.z));
}

inline __host__ __device__ void minCorner(float4 v, float3 u, float3 &ret){
    ret = make_float3(min(v.x, u.x),
                      min(v.y, u.y),
                      min(v.z, u.z));
}

inline __host__ __device__ float3 max(float3 v, float3 u){
    return make_float3(max(v.x, u.x),
                       max(v.y, u.y),
                       max(v.z, u.z));
}

inline __host__ __device__ float3 min(float3 v, float3 u){
    return make_float3(min(v.x, u.x),
                       min(v.y, u.y),
                       min(v.z, u.z));
}

#endif // _CUDA_PHOTON_UTILS_H_
