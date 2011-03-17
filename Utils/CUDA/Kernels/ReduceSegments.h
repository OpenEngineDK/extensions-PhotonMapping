// Kernels for segmenting triangle upper nodes
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

/*
#include <Meta/CUDA.h>
#include <Utils/CUDA/Kernels/DeviceVars.h>
#include <Utils/CUDA/Segments.h>
#include <Scene/KDNode.h>

namespace OpenEngine {
    using namespace Scene;
namespace Utils {
namespace CUDA {
namespace Kernels {
*/

#include <Utils/CUDA/SharedMemory.h>
#include <Utils/CUDA/Utils.h>

__device__ void fminf3(volatile float *v, volatile float *u, volatile float *ret){
    ret[0] = min(v[0], u[0]);
    ret[1] = min(v[1], u[1]);
    ret[2] = min(v[2], u[2]);
}

__device__ void fmaxf3(volatile float *v, volatile float *u, volatile float *ret){
    ret[0] = max(v[0], u[0]);
    ret[1] = max(v[1], u[1]);
    ret[2] = max(v[2], u[2]);
}


__device__ void reduceFminf3(volatile float *v, volatile float *u, volatile float *ret){
    ret[0] = v[0] = min(v[0], u[0]);
    ret[1] = v[1] = min(v[1], u[1]);
    ret[2] = v[2] = min(v[2], u[2]);
}

__device__ void reduceFmaxf3(volatile float *v, volatile float *u, volatile float *ret){
    ret[0] = v[0] = max(v[0], u[0]);
    ret[1] = v[1] = max(v[1], u[1]);
    ret[2] = v[2] = max(v[2], u[2]);
}


// @TODO write owner into the .w component? .o?

__global__ void
__launch_bounds__(Segments::SEGMENT_SIZE) 
    ReduceSegmentsNaive(int2 *primInfo,
                        float4 *aabbMin, float4 *aabbMax,
                        float4 *minResult, float4 *maxResult){
    
    const int segmentID = blockIdx.x;
        
    if (segmentID < d_segments){
        const int2 primitiveInfo = primInfo[segmentID];
        
        // Reduce!
        int offset = 1;
        while (threadIdx.x + offset < primitiveInfo.y){
            if (threadIdx.x % offset*2 == 0){
                float4 newV = aabbMin[threadIdx.x + primitiveInfo.x + offset];
                aabbMin[threadIdx.x + primitiveInfo.x] = min(aabbMin[threadIdx.x + primitiveInfo.x], newV);
                newV = aabbMax[threadIdx.x + primitiveInfo.x + offset];
                aabbMax[threadIdx.x + primitiveInfo.x] = max(aabbMax[threadIdx.x + primitiveInfo.x], newV);
            }else
                return;
            offset *= 2;
            __syncthreads();
        }

        if (threadIdx.x == 0){
            minResult[segmentID] = aabbMin[primitiveInfo.x];
            maxResult[segmentID] = aabbMax[primitiveInfo.x];
        }
    }
}    


__global__ void
__launch_bounds__(Segments::SEGMENT_SIZE) 
    ReduceSegmentsCoalesced(int2 *primInfo,
                            float4 *aabbMin, float4 *aabbMax,
                            float4 *minResult, float4 *maxResult){
    
    const int segmentID = blockIdx.x;
        
    if (segmentID < d_segments){
        const int2 primitiveInfo = primInfo[segmentID];
        
        // Reduce!
        for (int offset = NextPow2(primitiveInfo.y)/2; offset > 0; offset /= 2){
            if (threadIdx.x < offset){
                float4 newV = threadIdx.x + offset < primitiveInfo.y ? aabbMin[threadIdx.x + primitiveInfo.x + offset] : make_float4(fInfinity);
                aabbMin[threadIdx.x + primitiveInfo.x] = min(aabbMin[threadIdx.x + primitiveInfo.x], newV);
                newV = threadIdx.x + offset < primitiveInfo.y ? aabbMax[threadIdx.x + primitiveInfo.x + offset] : make_float4(-1.0f * fInfinity);
                aabbMax[threadIdx.x + primitiveInfo.x] = max(aabbMax[threadIdx.x + primitiveInfo.x], newV);
            }else
                return;
            __syncthreads();
        }

        if (threadIdx.x == 0){
            minResult[segmentID] = aabbMin[primitiveInfo.x];
            maxResult[segmentID] = aabbMax[primitiveInfo.x];
        }
    }
}    


__global__ void
__launch_bounds__(Segments::SEGMENT_SIZE) 
    ReduceSegmentsShared(int2 *primInfo,
                         float4 *aabbMin, float4 *aabbMax,
                         float4 *minResult, float4 *maxResult){
    
    const int segmentID = blockIdx.x;
        
    if (segmentID < d_segments){
        const int2 primitiveInfo = primInfo[segmentID];
        
        volatile __shared__ float sharedMin[3 * Segments::SEGMENT_SIZE];
        volatile __shared__ float sharedMax[3 * Segments::SEGMENT_SIZE];

        float3 global = threadIdx.x < primitiveInfo.y ? make_float3(aabbMin[threadIdx.x + primitiveInfo.x]) : make_float3(fInfinity);
        sharedMin[threadIdx.x * 3] = global.x;
        sharedMin[threadIdx.x * 3 + 1] = global.y;
        sharedMin[threadIdx.x * 3 + 2] = global.z;

        global = threadIdx.x < primitiveInfo.y ? make_float3(aabbMax[threadIdx.x + primitiveInfo.x]) : make_float3(-1.0f * fInfinity);
        sharedMax[threadIdx.x * 3] = global.x;
        sharedMax[threadIdx.x * 3 + 1] = global.y;
        sharedMax[threadIdx.x * 3 + 2] = global.z;

        __syncthreads();

        // Reduce!
        for (int offset = Segments::SEGMENT_SIZE/2; offset > 0; offset /= 2){
            if (threadIdx.x < offset){
                fminf3(sharedMin + threadIdx.x * 3, sharedMin + (threadIdx.x + offset) * 3, sharedMin + threadIdx.x * 3);
                fmaxf3(sharedMax + threadIdx.x * 3, sharedMax + (threadIdx.x + offset) * 3, sharedMax + threadIdx.x * 3);
            }else
                return;
            __syncthreads();
        }

        if (threadIdx.x == 0){
            float4 res;
            res.x = sharedMin[0];
            res.y = sharedMin[1];
            res.z = sharedMin[2];
            minResult[segmentID] = res;

            res.x = sharedMax[0];
            res.y = sharedMax[1];
            res.z = sharedMax[2];
            maxResult[segmentID] = res;
        }
    }
}    


__global__ void
__launch_bounds__(Segments::SEGMENT_SIZE) 
    ReduceSegmentsRegisters(int2 *primInfo,
                            float4 *aabbMin, float4 *aabbMax,
                            float4 *minResult, float4 *maxResult){
    
    const int segmentID = blockIdx.x;
        
    if (segmentID < d_segments){
        const int2 primitiveInfo = primInfo[segmentID];

        volatile __shared__ float sharedMin[3 * Segments::SEGMENT_SIZE];
        volatile __shared__ float sharedMax[3 * Segments::SEGMENT_SIZE];

        /* volatile */ float localMin[3];
        float3 global = threadIdx.x < primitiveInfo.y ? make_float3(aabbMin[threadIdx.x + primitiveInfo.x]) : make_float3(fInfinity);
        localMin[0] = sharedMin[threadIdx.x * 3] = global.x;
        localMin[1] = sharedMin[threadIdx.x * 3 + 1] = global.y;
        localMin[2] = sharedMin[threadIdx.x * 3 + 2] = global.z;

        /* volatile */ float localMax[3];
        global = threadIdx.x < primitiveInfo.y ? make_float3(aabbMax[threadIdx.x + primitiveInfo.x]) : make_float3(-1.0f * fInfinity);
        localMax[0] = sharedMax[threadIdx.x * 3] = global.x;
        localMax[1] = sharedMax[threadIdx.x * 3 + 1] = global.y;
        localMax[2] = sharedMax[threadIdx.x * 3 + 2] = global.z;

        __syncthreads();

        // Reduce!
        for (int offset = Segments::SEGMENT_SIZE/2; offset > 0; offset /= 2){
            if (threadIdx.x < offset){
                reduceFminf3(localMin, sharedMin + (threadIdx.x + offset) * 3, sharedMin + threadIdx.x * 3);
                reduceFmaxf3(localMax, sharedMax + (threadIdx.x + offset) * 3, sharedMax + threadIdx.x * 3);
            }else
                return;
            __syncthreads();
        }

        if (threadIdx.x == 0){
            float4 res;
            res.x = localMin[0];
            res.y = localMin[1];
            res.z = localMin[2];
            minResult[segmentID] = res;

            res.x = localMax[0];
            res.y = localMax[1];
            res.z = localMax[2];
            maxResult[segmentID] = res;
        }
    }
}    


/**
 * Switch to 3 floats, last one is useless.
 * Bank conflics? None as far as I can see in the profiler
 * Unroll. Decresed branching and instructions.
 * Do first
 */    
__global__ void
__launch_bounds__(Segments::SEGMENT_SIZE) 
    ReduceSegmentsUnrolled(int2 *primInfo,
                           float4 *aabbMin, float4 *aabbMax,
                           float4 *minResult, float4 *maxResult){
        
    const int segmentID = blockIdx.x;
        
    if (segmentID < d_segments){
        const int2 primitiveInfo = primInfo[segmentID];

        volatile __shared__ float sharedMin[3 * Segments::SEGMENT_SIZE/2];
        volatile __shared__ float sharedMax[3 * Segments::SEGMENT_SIZE/2];

        int offset = threadIdx.x + Segments::SEGMENT_SIZE / 2;
        /* volatile */ float localMin[3];
        float3 global = threadIdx.x < primitiveInfo.y ? make_float3(aabbMin[threadIdx.x + primitiveInfo.x]) : make_float3(fInfinity);
        global = min(global, offset < primitiveInfo.y ? make_float3(aabbMin[offset + primitiveInfo.x]) : make_float3(fInfinity));
        localMin[0] = sharedMin[threadIdx.x * 3] = global.x;
        localMin[1] = sharedMin[threadIdx.x * 3 + 1] = global.y;
        localMin[2] = sharedMin[threadIdx.x * 3 + 2] = global.z;

        /* volatile */ float localMax[3];
        global = threadIdx.x < primitiveInfo.y ? make_float3(aabbMax[threadIdx.x + primitiveInfo.x]) : make_float3(-1.0f * fInfinity);
        global = max(global, offset < primitiveInfo.y ? make_float3(aabbMax[offset + primitiveInfo.x]) : make_float3(-1.0f * fInfinity));
        localMax[0] = sharedMax[threadIdx.x * 3] = global.x;
        localMax[1] = sharedMax[threadIdx.x * 3 + 1] = global.y;
        localMax[2] = sharedMax[threadIdx.x * 3 + 2] = global.z;

        __syncthreads();

        // Reduce!
        /*
        for (int offset = Segments::SEGMENT_SIZE/4; offset > warpSize; offset /= 2){
            if (threadIdx.x < offset){
                reduceFminf3(localMin, sharedMin + (threadIdx.x + offset) * 3, sharedMin + threadIdx.x * 3);
                reduceFmaxf3(localMax, sharedMax + (threadIdx.x + offset) * 3, sharedMax + threadIdx.x * 3);
            }else
                return;
            __syncthreads();
        }
        */

        if (128 <= Segments::SEGMENT_SIZE/4){
            if (threadIdx.x < 128){
                reduceFminf3(localMin, sharedMin + (threadIdx.x + 128) * 3, sharedMin + threadIdx.x * 3);
                reduceFmaxf3(localMax, sharedMax + (threadIdx.x + 128) * 3, sharedMax + threadIdx.x * 3);
            }else
                return;
            __syncthreads();
        }

        if (64 <= Segments::SEGMENT_SIZE/4){
            if (threadIdx.x < 64){
                reduceFminf3(localMin, sharedMin + (threadIdx.x + 64) * 3, sharedMin + threadIdx.x * 3);
                reduceFmaxf3(localMax, sharedMax + (threadIdx.x + 64) * 3, sharedMax + threadIdx.x * 3);
            }else
                return;
            __syncthreads();
        }

        /*
        for (int offset = warpSize; offset > 1; offset /= 2){
            reduceFminf3(localMin, sharedMin + (threadIdx.x + offset) * 3, sharedMin + threadIdx.x * 3);
            reduceFmaxf3(localMax, sharedMax + (threadIdx.x + offset) * 3, sharedMax + threadIdx.x * 3);
        }
        */

        reduceFminf3(localMin, sharedMin + (threadIdx.x + 32) * 3, sharedMin + threadIdx.x * 3);
        reduceFmaxf3(localMax, sharedMax + (threadIdx.x + 32) * 3, sharedMax + threadIdx.x * 3);

        reduceFminf3(localMin, sharedMin + (threadIdx.x + 16) * 3, sharedMin + threadIdx.x * 3);
        reduceFmaxf3(localMax, sharedMax + (threadIdx.x + 16) * 3, sharedMax + threadIdx.x * 3);

        reduceFminf3(localMin, sharedMin + (threadIdx.x + 8) * 3, sharedMin + threadIdx.x * 3);
        reduceFmaxf3(localMax, sharedMax + (threadIdx.x + 8) * 3, sharedMax + threadIdx.x * 3);

        reduceFminf3(localMin, sharedMin + (threadIdx.x + 4) * 3, sharedMin + threadIdx.x * 3);
        reduceFmaxf3(localMax, sharedMax + (threadIdx.x + 4) * 3, sharedMax + threadIdx.x * 3);

        reduceFminf3(localMin, sharedMin + (threadIdx.x + 2) * 3, sharedMin + threadIdx.x * 3);
        reduceFmaxf3(localMax, sharedMax + (threadIdx.x + 2) * 3, sharedMax + threadIdx.x * 3);

        if (threadIdx.x == 0){
            float4 res;
            res.x = min(localMin[0], sharedMin[3]);
            res.y = min(localMin[1], sharedMin[4]);
            res.z = min(localMin[2], sharedMin[5]);
            minResult[segmentID] = res;

            res.x = max(localMax[0], sharedMax[3]);
            res.y = max(localMax[1], sharedMax[4]);
            res.z = max(localMax[2], sharedMax[5]);
            maxResult[segmentID] = res;
        }
    }
}    

__global__ void AabbMemset(float4* aabbMin, float4* aabbMax){
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < d_activeNodeRange){
        aabbMin[id] = make_float4(fInfinity);
        aabbMax[id] = make_float4(-1.0f * fInfinity);
    }
}

__global__ void FinalSegmentedReduce(float4 *segmentAabbMin,
                                     float4 *segmentAabbMax,
                                     int *segmentOwner,
                                     float4 *nodeAabbMin,
                                     float4 *nodeAabbMax){
    
    const unsigned int segmentID = threadIdx.x;

    int index0 = segmentID * 2;
    int index1 = index0 + 1;
    
    while (index1 < d_segments){
        int owner0 = segmentOwner[index0];
        int owner1 = segmentOwner[index1];
        
        if (owner0 != owner1){
            owner1 -= d_activeNodeIndex;
            nodeAabbMin[owner1] = min(nodeAabbMin[owner1], segmentAabbMin[index1]);
            nodeAabbMax[owner1] = max(nodeAabbMax[owner1], segmentAabbMax[index1]);
        }else{
            segmentAabbMin[index0] = min(segmentAabbMin[index0], segmentAabbMin[index1]);
            segmentAabbMax[index0] = max(segmentAabbMax[index0], segmentAabbMax[index1]);
        }
        
        index0 *= 2;
        index1 *= 2;
        __syncthreads();
    }

    if (threadIdx.x == 0){
        int owner = segmentOwner[0] - d_activeNodeIndex;
        nodeAabbMin[owner] = min(nodeAabbMin[owner], segmentAabbMin[0]);
        nodeAabbMax[owner] = max(nodeAabbMax[owner], segmentAabbMax[0]);
    }
}

#define MAX_SEGS_PR_KERNEL 4

__global__ void
__launch_bounds__(MAX_SEGS_PR_KERNEL) 
    InitializeHat(float4 *resMin, float4 *resMax, int segments){

    const int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < segments){
        resMin[id] = make_float4(fInfinity);
        resMax[id] = make_float4(-1.0f * fInfinity);
    }
}

__global__ void
__launch_bounds__(MAX_SEGS_PR_KERNEL) 
    TestingHat(float4 *inMin, float4 *inMax, int* inOwner,
               float4 *resMin, float4* resMax,
               float4* tempMin, float4* tempMax, int* tempOwner,
               int segments){

    const int startIndex = blockIdx.x * MAX_SEGS_PR_KERNEL;    
    int range = segments - startIndex;
    range = range > MAX_SEGS_PR_KERNEL ? MAX_SEGS_PR_KERNEL : range;

    int ownerStart = inOwner[startIndex];
    int ownerRange = inOwner[startIndex + range-1] - ownerStart + 1;

    __shared__ int owner[MAX_SEGS_PR_KERNEL];
    volatile __shared__ float sMin[3*MAX_SEGS_PR_KERNEL];
    volatile __shared__ float sMax[3*MAX_SEGS_PR_KERNEL];
    if (threadIdx.x < range){
        owner[threadIdx.x] = inOwner[threadIdx.x + startIndex] - ownerStart;
        float3 global = make_float3(inMin[threadIdx.x + startIndex]);
        sMin[3*threadIdx.x] = global.x; sMin[3*threadIdx.x+1] = global.y; sMin[3*threadIdx.x+2] = global.z;
        global = make_float3(inMax[threadIdx.x + startIndex]);
        sMax[3*threadIdx.x] = global.x; sMax[3*threadIdx.x+1] = global.y; sMax[3*threadIdx.x+2] = global.z;
    }

    volatile __shared__ float rMin[3*MAX_SEGS_PR_KERNEL];
    volatile __shared__ float rMax[3*MAX_SEGS_PR_KERNEL];
    if (threadIdx.x < ownerRange){
        float3 res = make_float3(resMin[ownerStart + threadIdx.x]);
        rMin[3*threadIdx.x] = res.x; rMin[3*threadIdx.x+1] = res.y; rMin[3*threadIdx.x+2] = res.z;
        res = make_float3(resMax[ownerStart + threadIdx.x]);
        rMax[3*threadIdx.x] = res.x; rMax[3*threadIdx.x+1] = res.y; rMax[3*threadIdx.x+2] = res.z;
    }
    __syncthreads();
    
    int index0 = threadIdx.x * 2;
    int index1 = index0+1;
    while(index1 < range){
        int owner0 = owner[index0];
        int owner1 = owner[index1];
        
        if (owner0 == owner1){
            sMin[3*index0] = min(sMin[3*index0], sMin[3*index1]);
            sMin[3*index0+1] = min(sMin[3*index0+1], sMin[3*index1+1]);
            sMin[3*index0+2] = min(sMin[3*index0+2], sMin[3*index1+2]);
            sMax[3*index0] = max(sMax[3*index0], sMax[3*index1]);
            sMax[3*index0+1] = max(sMax[3*index0+1], sMax[3*index1+1]);
            sMax[3*index0+2] = max(sMax[3*index0+2], sMax[3*index1+2]);
        }else{
            rMin[3*owner1] = min(rMin[3*owner1], sMin[3*index1]);
            rMin[3*owner1+1] = min(rMin[3*owner1+1], sMin[3*index1+1]);
            rMin[3*owner1+2] = min(rMin[3*owner1+2], sMin[3*index1+2]);
            rMax[3*owner1] = max(rMax[3*owner1], sMax[3*index1]);
            rMax[3*owner1+1] = max(rMax[3*owner1+1], sMax[3*index1+1]);
            rMax[3*owner1+2] = max(rMax[3*owner1+2], sMax[3*index1+2]);
        }
        
        index0 *= 2;
        index1 *= 2;
        __syncthreads();
    }

    // Dump results
    if (threadIdx.x != 0 && threadIdx.x < ownerRange){
        resMin[ownerStart + threadIdx.x] = make_float4(rMin[3*threadIdx.x], 
                                                       rMin[3*threadIdx.x+1],
                                                       rMin[3*threadIdx.x+2], 0);
        resMax[ownerStart + threadIdx.x] = make_float4(rMax[3*threadIdx.x],
                                                       rMax[3*threadIdx.x+1],
                                                       rMax[3*threadIdx.x+2], 0);
    }

    if (threadIdx.x == 0){
        tempMin[blockIdx.x] = make_float4(sMin[0], sMin[1], sMin[2], 0);
        tempMax[blockIdx.x] = make_float4(sMax[0], sMax[1], sMax[2], 0);
        tempOwner[blockIdx.x] = ownerStart;
    }
}

__global__ void CalcUpperNodeSplitInfo(float4 *aabbMin, float4* aabbMax,
                                       float *splitPos, char *info){
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
        
    if (id < d_activeNodeRange){
            
        float3 bbSize = make_float3(aabbMax[id]) - make_float3(aabbMin[id]);
        float3 median = bbSize * 0.5 + make_float3(aabbMin[id]);

        // Calculate splitting plane
        bool yAboveX = bbSize.x < bbSize.y;
        float max = yAboveX ? bbSize.y : bbSize.x;
        float split = yAboveX ? median.y : median.x;
        char axis = yAboveX ? KDNode::Y : KDNode::X;
        bool zHigher = max < bbSize.z;
        splitPos[id] = zHigher ? median.z : split;
        info[id] = zHigher ? KDNode::Z : axis;
    }
}
