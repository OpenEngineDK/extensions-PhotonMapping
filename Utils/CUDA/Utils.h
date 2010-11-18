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

#if OE_SAFE
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

inline bool Calc1DKernelDimensions(const unsigned int size, 
                                   unsigned int &blocks, unsigned int &threads,
                                   unsigned int maxThreads = 0){
    const unsigned int MAX_THREADS = maxThreads ? maxThreads : activeCudaDevice.maxThreadsDim[0];
    const unsigned int MAX_BLOCKS = activeCudaDevice.maxGridSize[0];

    if (size < MAX_THREADS){
        threads = size;
        blocks = 1;
        return true;
    }else{
        threads = MAX_THREADS;
        blocks = ((size+1) / MAX_THREADS) + 1;
        return blocks > MAX_BLOCKS;
    }
}

inline bool Calc1DKernelDimensionsWithSmem(const unsigned int size, 
                                           unsigned int &blocks, unsigned int &threads, unsigned int &memSize,
                                           unsigned int maxThreads = 0){
    return false;
}

inline __host__ __device__ bool TriangleRayIntersection(float3 v0, float3 v1, float3 v2,
                                                        float3 origin, float3 direction,
                                                        float3 &hit){
    
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    float3 t = origin - v0;
    float3 p = cross(direction, e2);
    float3 q = cross(t, e1);
    float det = dot(p, e1);

    // if det is 'equal' to zero, the ray lies in the triangles plane
    // and cannot be seen.
    //if (det == 0.0f) return false;

    det = 1.0f / det;

    hit.x = det * dot(q, e2);
    hit.y = det * dot(p, t);
    hit.z = det * dot(q, direction);

    return hit.x >= 0.0f && hit.y >= 0.0f && hit.z >= 0.0f && hit.y + hit.z <= 1.0f;
}

inline __host__ __device__ int firstBitSet(int n){
#ifdef __CUDA_ARCH__ // device code
    return __ffs(n);
#else
    return ffs(n);
#endif    
}

/**
 * Stolen from
 * http://gurmeetsingh.wordpress.com/2008/08/05/fast-bit-counting-routines
 *
 * How and why it works? Magic! Now go code something.
 */
inline __host__ __device__ int bitcount(int n){
#ifdef __CUDA_ARCH__ // device code
    return __popc(n);
#else
    int tmp = n - ((n >> 1) & 033333333333)
        - ((n >> 2) & 011111111111);
    return ((tmp + (tmp >> 3)) & 030707070707) % 63;
#endif
}

inline std::string BitmapToString(unsigned int n){
    std::ostringstream out;
    out << "[";
    for (unsigned int i = 0; i < 31; ++i){
        if (n & 1<<i)
            out << 1 << ", ";
        else
            out << 0 << ", ";
    }
    if (n & (unsigned int)1<<31)
        out << 1 << "]";
    else
        out << 0 << "]";
    
    return out.str();
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

inline __host__ __device__ float4 max(float4 v, float4 u){
    return make_float4(max(v.x, u.x),
                       max(v.y, u.y),
                       max(v.z, u.z),
                       max(v.w, u.w));
}

inline __host__ __device__ float4 min(float4 v, float4 u){
    return make_float4(min(v.x, u.x),
                       min(v.y, u.y),
                       min(v.z, u.z),
                       min(v.w, u.w));
}

#endif // _CUDA_PHOTON_UTILS_H_
