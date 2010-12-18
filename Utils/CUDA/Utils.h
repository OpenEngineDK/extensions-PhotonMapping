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
#define PRINT_TIMER(timerID, name)                                      \
    do{                                                                 \
        cudaThreadSynchronize();                                        \
        cutStopTimer(timerID);                                          \
        logger.info << name << " time: " << cutGetTimerValue(timerID) << "ms" << logger.end; \
    }while(false)                                                       \

#ifdef __CUDA_ARCH__
#define CUDALogger(string)
#else
#define CUDALogger(string) logger.info << string << logger.end
#endif

template <class T>
inline __host__ __device__
T FetchDeviceData(T &symbol){
#ifdef __CUDA_ARCH__
    return symbol;
#else
    T ret;
    cudaMemcpyFromSymbol(&ret, symbol, sizeof(T));
    return ret;
#endif
}
  
template <class T>
inline __host__ __device__
T FetchGlobalData(const T* a, const int i){
#ifdef __CUDA_ARCH__
    return a[i];
#else
    T ret;
    cudaMemcpy(&ret, a + i, sizeof(T), cudaMemcpyDeviceToHost);
    return ret;
#endif
}

template <class T>
inline __host__ __device__
void DumpGlobalData(const T d, T* a, const int i){
#ifdef __CUDA_ARCH__
    a[i] = d;
#else
    cudaMemcpy(a + i, &d, sizeof(T), cudaMemcpyHostToDevice);
#endif
}

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

inline __host__ __device__ 
float4 make_float4(float x, float y, float z){
    float4 r;
    r.x = x; r.y = y; r.z = z;
    return r;
}

#endif // _CUDA_PHOTON_UTILS_H_
