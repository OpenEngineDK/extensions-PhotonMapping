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
#include <stdint.h>

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

struct KernelConf {
    int blocks;
    int threads;
    int smem;
};

inline KernelConf KernelConf1D(const unsigned int threads, unsigned int maxThreadsPrBlock = 2<<16,
                               const unsigned int registerUsage = 0,
                               const unsigned int smemPrThread = 0){

    KernelConf conf;
    unsigned int maxThreads = min(maxThreadsPrBlock, activeCudaDevice.maxThreadsDim[0]);
    
    // Take register usage into account
    if (registerUsage) maxThreads = min(maxThreadsPrBlock, activeCudaDevice.regsPerBlock / registerUsage);

    // Take shared mem into account
    if (smemPrThread) maxThreads = min(maxThreadsPrBlock, (unsigned int)activeCudaDevice.sharedMemPerBlock / smemPrThread);
    
    if (threads < maxThreads){
        conf.threads = ((threads - 1) / activeCudaDevice.warpSize + 1) * activeCudaDevice.warpSize;
        conf.blocks = 1;
    }else{
        conf.threads = maxThreads;
        conf.blocks = min(((threads-1) / maxThreads) + 1, activeCudaDevice.maxGridSize[0]);
    }

    conf.smem = smemPrThread * conf.threads;

    return conf;
}

inline bool Calc1DKernelDimensions(const unsigned int size, 
                                   unsigned int &blocks, unsigned int &threads,
                                   unsigned int maxThreads = 0){
    const unsigned int MAX_THREADS = maxThreads ? maxThreads : activeCudaDevice.maxThreadsDim[0];
    const unsigned int MAX_BLOCKS = activeCudaDevice.maxGridSize[0];

    if (size < MAX_THREADS){
        threads = ((size - 1) / activeCudaDevice.warpSize + 1) * activeCudaDevice.warpSize;
        blocks = 1;
        return true;
    }else{
        threads = MAX_THREADS;
        blocks = ((size-1) / MAX_THREADS) + 1;
        return blocks > MAX_BLOCKS;
    }
}

inline bool Calc1DKernelDimensionsWithSmem(const unsigned int size, const unsigned int smemPrThread,
                                           unsigned int &blocks, unsigned int &threads, unsigned int &smemSize,
                                           unsigned int maxThreads = 0){

    const unsigned int MAX_THREADS = maxThreads ? maxThreads : activeCudaDevice.maxThreadsDim[0];
    const unsigned int smemThreads = activeCudaDevice.sharedMemPerBlock / smemPrThread;
    
    maxThreads = MAX_THREADS < smemThreads ? MAX_THREADS : smemThreads;

    bool succes = Calc1DKernelDimensions(size, blocks, threads, maxThreads);

    smemSize = threads * smemPrThread;

    return succes;
}

inline __host__ __device__ bool TriangleRayIntersection(const float3 v0, const float3 v1, const float3 v2,
                                                        const float3 origin, const float3 direction,
                                                        float3 &hit){
    
    const float3 e1 = v1 - v0;
    const float3 e2 = v2 - v0;
    const float3 t = origin - v0;
    const float3 p = cross(direction, e2);
    const float3 q = cross(t, e1);
    const float det = dot(p, e1);

    // if det is 'equal' to zero, the ray lies in the triangles plane
    // and cannot be seen.
    //if (det == 0.0f) return false;

    const float invDet = 1.0f / det;

    hit.x = invDet * dot(q, e2);
    hit.y = invDet * dot(p, t);
    hit.z = invDet * dot(q, direction);

    return hit.x >= 0.0f && hit.y >= 0.0f && hit.z >= 0.0f && hit.y + hit.z <= 1.0f;
}

inline __host__ __device__ bool TriangleAabbIntersection(float3 v0, float3 v1, float3 v2, 
                                                  const float3 aabbMin, const float3 aabbMax){

    const float3 f0 = v1 - v0;
    const float3 f1 = v2 - v1;
    const float3 f2 = v0 - v2;

    const float3 halfSize = (aabbMax - aabbMin) * 0.5f;
    const float3 center = aabbMin + halfSize;

    v0 -= center; v1 -= center; v2 -= center;

    // Only test 3 from Akenine-Möller
    
    // a00
    float p0 = v0.z * v1.y - v0.y * v1.z;
    float p1 = (v1.y - v0.y) * v2.z - (v1.z - v0.z) * v2.y;
    float r = halfSize.y * fabsf(f0.z) + halfSize.z * fabsf(f0.y);
    bool res = !(p0 > r || p1 > r || p0 < -r || p1 < -r);

    // a01
    p0 = v1.z * v2.y - v1.y * v2.z;
    p1 = (v2.y - v1.y) * v0.z - (v2.z - v1.z) * v0.y;
    r = halfSize.y * fabsf(f1.z) + halfSize.z * fabsf(f1.y);
    res &= !((p0 > r && p1 > r) || (p0 < -r && p1 < -r));

    // a02
    p0 = v2.z * v0.y - v2.y * v0.z;
    p1 = (v0.y - v2.y) * v1.z - (v0.z - v2.z) * v1.y;
    r = halfSize.y * fabsf(f2.z) + halfSize.z * fabsf(f2.y);
    res &= !((p0 > r && p1 > r) || (p0 < -r && p1 < -r));

    // a10
    p0 = v0.x * v1.z - v0.z * v1.x;
    p1 = (v1.z - v0.z) * v2.x - (v1.x - v0.x) * v2.z;
    r = halfSize.x * fabsf(f0.z) + halfSize.z * fabsf(f0.x);
    res &= !((p0 > r && p1 > r) || (p0 < -r && p1 < -r));

    // a11
    p0 = v1.x * v2.z - v1.z * v2.x;
    p1 = (v2.z - v1.z) * v0.x - (v2.x - v1.x) * v0.z;
    r = halfSize.x * fabsf(f1.z) + halfSize.z * fabsf(f1.x);
    res &= !((p0 > r && p1 > r) || (p0 < -r && p1 < -r));

    // a12
    p0 = v2.x * v0.z - v2.z * v0.x;
    p1 = (v0.z - v2.z) * v1.x - (v0.x - v2.x) * v1.z;
    r = halfSize.x * fabsf(f2.z) + halfSize.z * fabsf(f2.x);
    res &= !((p0 > r && p1 > r) || (p0 < -r && p1 < -r));

    // a20
    p0 = v0.y * v1.x - v0.x * v1.y;
    p1 = (v1.x - v0.x) * v2.y - (v1.y - v0.y) * v2.x;
    r = halfSize.x * fabsf(f0.y) + halfSize.y * fabsf(f0.x);
    res &= !((p0 > r && p1 > r) || (p0 < -r && p1 < -r));

    // a21
    p0 = v1.y * v2.x - v1.x * v2.y;
    p1 = (v2.x - v1.x) * v0.y - (v2.y - v1.y) * v0.x;
    r = halfSize.x * fabsf(f1.y) + halfSize.y * fabsf(f1.x);
    res &= !((p0 > r && p1 > r) || (p0 < -r && p1 < -r));

    // a22
    p0 = v2.y * v0.x - v2.x * v0.y;
    p1 = (v0.x - v2.x) * v1.y - (v0.y - v2.y) * v1.x;
    r = halfSize.x * fabsf(f2.y) + halfSize.y * fabsf(f2.x);
    res &= !((p0 > r && p1 > r) || (p0 < -r && p1 < -r));

    return res;
}

inline __host__ __device__ int firstBitSet(const int n){
#ifdef __CUDA_ARCH__ // device code
    return __ffs(n);
#else
    return ffs(n);
#endif    
}

inline __host__ __device__ int firstBitSet(const long long n){
#ifdef __CUDA_ARCH__ // device code
    return __ffsll(n);
#else
    int ffs = firstBitSet(int(n));
    if (ffs) return ffs;
    ffs = firstBitSet(int(n>>32));
    if (ffs) return 32 + ffs;
    return 0;
#endif
}

/**
 * Stolen from
 * http://gurmeetsingh.wordpress.com/2008/08/05/fast-bit-counting-routines
 *
 * How and why it works? Magic! Now go code something.
 */
inline __host__ __device__ int bitcount(const int n){
#ifdef __CUDA_ARCH__ // device code
    return __popc(n);
#else
    int tmp = n - ((n >> 1) & 033333333333)
        - ((n >> 2) & 011111111111);
    return ((tmp + (tmp >> 3)) & 030707070707) % 63;
#endif
}

inline __host__ __device__ int bitcount(const long long int n){
#ifdef __CUDA_ARCH__ // device code
    return __popcll(n);
#else
    return bitcount(int(n)) + bitcount(int(n>>32));
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
inline std::string BitmapToString(int n){
    return BitmapToString((unsigned int)n);
}


inline std::string BitmapToString(long long int n){
    std::ostringstream out;
    out << "[";
    for (unsigned int i = 0; i < 63; ++i){
        if (n & (long long int)1<<i)
            out << 1 << ", ";
        else
            out << 0 << ", ";
    }
    if (n & (unsigned long long int)1<<63)
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
