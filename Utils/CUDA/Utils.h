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
                                   unsigned int &blocks, unsigned int &threads){
    const unsigned int MAX_THREADS = activeCudaDevice.maxThreadsDim[0];
    const unsigned int MAX_BLOCKS = activeCudaDevice.maxGridSize[0];

    unsigned int s = NextPow2(size);
    threads = (s < MAX_THREADS) ? s : MAX_THREADS;
    //blocks = (size + (threads * 2 - 1)) / (threads * 2);
    //blocks = min(MAX_BLOCKS, blocks);
    s /= threads;
    blocks = s < MAX_BLOCKS ? s : MAX_BLOCKS;
}

inline void Calc2DKernelDimensions(const int size, 
                                   dim3 &blocks, dim3 &threads){
    const int2 MAX_THREADS = make_int2(activeCudaDevice.maxThreadsDim[0],
                                       activeCudaDevice.maxThreadsDim[1]);

    const int2 MAX_BLOCKS = make_int2((activeCudaDevice.maxGridSize[0]+1) / MAX_THREADS.x,
                                      (activeCudaDevice.maxGridSize[1]+1) / MAX_THREADS.y);

    int s = NextPow2(size);
    threads.x = s > MAX_THREADS.x ? MAX_THREADS.x : s;
    s /= threads.x;

    blocks.x = s > MAX_BLOCKS.x ? MAX_BLOCKS.x : s;
    s /= blocks.x;
    blocks.y = s > MAX_BLOCKS.y ? MAX_BLOCKS.y : s;

    threads.y = threads.z = blocks.z = 1;
}

inline __host__ __device__ int globalIdx2D(const uint3 threadIdx, const uint3 blockIdx, 
                                           const dim3 blockDim, const dim3 gridDim){
    int blockSize = blockDim.x * blockDim.y;
    int localIdx = threadIdx.x;// + blockDim.x * threadIdx.y;
    int blockId = blockIdx.x + gridDim.x * blockIdx.y;
    return localIdx + blockSize * blockId;
}

/*
inline __device__ int globalIdx2D(){
    int blockSize = blockDim.x * blockDim.y;
    int localIdx = threadIdx.x + blockDim.x * threadIdx.y;
    int blockId = blockIdx.x + gridDim.x * blockIdx.y;
    return localIdx + blockSize * blockId;
}
*/

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
