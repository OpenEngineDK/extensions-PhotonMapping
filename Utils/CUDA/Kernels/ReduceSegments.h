// Kernels for segmenting triangle upper nodes
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Meta/CUDA.h>
#include <Utils/CUDA/Kernels/PhotonMapDeviceVars.h>
#include <Utils/CUDA/Segments.h>
#include <Utils/CUDA/SharedMemory.h>
#include <Utils/CUDA/Utils.h>
#include <Scene/KDNode.h>

namespace OpenEngine {
    using namespace Scene;
namespace Utils {
namespace CUDA {
namespace Kernels {

    /**
     * Optimizations yield ~50%
     *
     * Remove unnecessary syncthreads.
     * Use local memory to avoid a shared memory lookup.
     * Unroll final reduction into the output step.
     * Explicitly kill threads that will no longer do anything useful.
     *
     * PENDING
     * Do a complete unroll, since the freaking compiler can't!!
     * use float3 in shared mem
     */
    __global__ void 
    __launch_bounds__(Segments::SEGMENT_SIZE) 
        ReduceSegments(int2 *primInfo,
                       float4 *aabbMin, float4 *aabbMax,
                       float4 *minResult, float4 *maxResult){
        
        const int segmentID = blockIdx.x;

        if (segmentID < d_segments){
            float4* sharedMin = SharedMemory<float4>();
            float4* sharedMax = sharedMin + Segments::SEGMENT_SIZE;
            
            const int2 primitiveInfo = primInfo[segmentID];

            float4 localMin = sharedMin[threadIdx.x] = threadIdx.x < primitiveInfo.y ? aabbMin[threadIdx.x + primitiveInfo.x] : make_float4(fInfinity);
            float4 localMax = sharedMax[threadIdx.x] = threadIdx.x < primitiveInfo.y ? aabbMax[threadIdx.x + primitiveInfo.x] : make_float4(-1.0f * fInfinity);
            __syncthreads();
            
            // Reduce!
            for (int offset = Segments::SEGMENT_SIZE/2; offset > warpSize; offset /= 2){
                if (threadIdx.x < offset){
                    sharedMin[threadIdx.x] = localMin = min(localMin, sharedMin[threadIdx.x + offset]);
                    sharedMax[threadIdx.x] = localMax = max(localMax, sharedMax[threadIdx.x + offset]);
                }else
                    return;
                __syncthreads();
            }
            
            for (int offset = warpSize; offset > 1; offset /= 2){
                sharedMin[threadIdx.x] = localMin = min(localMin, sharedMin[threadIdx.x + offset]);
                sharedMax[threadIdx.x] = localMax = max(localMax, sharedMax[threadIdx.x + offset]);
            }

            if (threadIdx.x == 0){
                minResult[segmentID] = min(localMin, sharedMin[1]);
                maxResult[segmentID] = max(localMax, sharedMax[1]);
            }
        }
    }

    __device__ void fminf4(volatile float *v, volatile float *u, volatile float *ret){
        ret[0] = min(v[0], u[0]);
        ret[1] = min(v[1], u[1]);
        ret[2] = min(v[2], u[2]);
        ret[3] = min(v[3], u[3]);
    }

    __device__ void fmaxf4(volatile float *v, volatile float *u, volatile float *ret){
        ret[0] = max(v[0], u[0]);
        ret[1] = max(v[1], u[1]);
        ret[2] = max(v[2], u[2]);
        ret[3] = max(v[3], u[3]);
    }
    
    __global__ void
    __launch_bounds__(Segments::SEGMENT_SIZE) 
        ReduceSegmentsShared(int2 *primInfo,
                             float4 *aabbMin, float4 *aabbMax,
                             float4 *minResult, float4 *maxResult){
        
        const int segmentID = blockIdx.x;
        
        if (segmentID < d_segments){
            const int2 primitiveInfo = primInfo[segmentID];

            volatile float* sharedMin = SharedMemory<float>();
            volatile float* sharedMax = sharedMin + Segments::SEGMENT_SIZE * 4;

            volatile float localMin[4];
            volatile float localMax[4];
            float4 global = threadIdx.x < primitiveInfo.y ? aabbMin[threadIdx.x + primitiveInfo.x] : make_float4(fInfinity);
            localMin[0] = sharedMin[threadIdx.x * 4] = global.x;
            localMin[1] = sharedMin[threadIdx.x * 4 + 1] = global.y;
            localMin[2] = sharedMin[threadIdx.x * 4 + 2] = global.z;
            localMin[3] = sharedMin[threadIdx.x * 4 + 3] = global.w;

            global = threadIdx.x < primitiveInfo.y ? aabbMax[threadIdx.x + primitiveInfo.x] : make_float4(-1.0f * fInfinity);
            localMax[0] = sharedMax[threadIdx.x * 4] = global.x;
            localMax[1] = sharedMax[threadIdx.x * 4 + 1] = global.y;
            localMax[2] = sharedMax[threadIdx.x * 4 + 2] = global.z;
            localMax[3] = sharedMax[threadIdx.x * 4 + 3] = global.w;

            __syncthreads();

            // Reduce!
            for (int offset = Segments::SEGMENT_SIZE/2; offset > warpSize; offset /= 2){
                if (threadIdx.x < offset){
                    //sharedMin[threadIdx.x] = localMin = min(localMin, sharedMin[threadIdx.x + offset]);
                    fminf4(localMin, sharedMin + (threadIdx.x + offset) * 4, localMin);
                    sharedMin[threadIdx.x * 4] = localMin[0];
                    sharedMin[threadIdx.x * 4 + 1] = localMin[1];
                    sharedMin[threadIdx.x * 4 + 2] = localMin[2];
                    sharedMin[threadIdx.x * 4 + 3] = localMin[3];

                    //sharedMax[threadIdx.x] = localMax = max(localMax, sharedMax[threadIdx.x + offset]);
                    fmaxf4(localMax, sharedMax + (threadIdx.x + offset) * 4, localMax);
                    sharedMax[threadIdx.x * 4] = localMax[0];
                    sharedMax[threadIdx.x * 4 + 1] = localMax[1];
                    sharedMax[threadIdx.x * 4 + 2] = localMax[2];
                    sharedMax[threadIdx.x * 4 + 3] = localMax[3];
                }else
                    return;
                __syncthreads();
            }

            for (int offset = warpSize; offset > 1; offset /= 2){
                //sharedMin[threadIdx.x] = localMin = min(localMin, sharedMin[threadIdx.x + offset]);
                fminf4(localMin, sharedMin + (threadIdx.x + offset) * 4, localMin);
                sharedMin[threadIdx.x * 4] = localMin[0];
                sharedMin[threadIdx.x * 4 + 1] = localMin[1];
                sharedMin[threadIdx.x * 4 + 2] = localMin[2];
                sharedMin[threadIdx.x * 4 + 3] = localMin[3];
                
                //sharedMax[threadIdx.x] = localMax = max(localMax, sharedMax[threadIdx.x + offset]);
                fmaxf4(localMax, sharedMax + (threadIdx.x + offset) * 4, localMax);
                sharedMax[threadIdx.x * 4] = localMax[0];
                sharedMax[threadIdx.x * 4 + 1] = localMax[1];
                sharedMax[threadIdx.x * 4 + 2] = localMax[2];
                sharedMax[threadIdx.x * 4 + 3] = localMax[3];
                __syncthreads();
            }

            if (threadIdx.x == 0){
                //minResult[segmentID] = min(make_float4(localMin), make_float4(sharedMin+4));
                float4 res;
                res.x = min(localMin[0], sharedMin[4]);
                res.y = min(localMin[1], sharedMin[5]);
                res.z = min(localMin[2], sharedMin[6]);
                res.w = min(localMin[3], sharedMin[7]);
                minResult[segmentID] = res;

                //maxResult[segmentID] = max(make_float4(localMax), make_float4(sharedMax+4));
                res.x = max(localMax[0], sharedMax[4]);
                res.y = max(localMax[1], sharedMax[5]);
                res.z = max(localMax[2], sharedMax[6]);
                res.w = max(localMax[3], sharedMax[7]);
                maxResult[segmentID] = res;
            }
        }
    }    

    __global__ void 
    __launch_bounds__(Segments::SEGMENT_SIZE) 
        GlobalReduceSegments(int2 *primInfo,
                             float4 *aabbMin, float4 *aabbMax,
                             float4 *minResult, float4 *maxResult){
        
        const int segmentID = blockIdx.x;

        if (segmentID < d_segments){
            const int2 primitiveInfo = primInfo[segmentID];

            // Reduce!
            for (int offset = Segments::SEGMENT_SIZE/2; offset > warpSize; offset /= 2){
                if (threadIdx.x < offset){
                    aabbMin[threadIdx.x + primitiveInfo.x] = min(aabbMin[threadIdx.x + primitiveInfo.x], 
                                                                 aabbMin[threadIdx.x + offset + primitiveInfo.x]);
                    aabbMax[threadIdx.x + primitiveInfo.x] = max(aabbMax[threadIdx.x + primitiveInfo.x], 
                                                                 aabbMax[threadIdx.x + offset + primitiveInfo.x]);
                }else
                    return;
                __syncthreads();
            }
            
            for (int offset = warpSize; offset > 1; offset /= 2){
                aabbMin[threadIdx.x + primitiveInfo.x] = min(aabbMin[threadIdx.x + primitiveInfo.x], 
                                                             aabbMin[threadIdx.x + offset + primitiveInfo.x]);
                aabbMax[threadIdx.x + primitiveInfo.x] = max(aabbMax[threadIdx.x + primitiveInfo.x], 
                                                             aabbMax[threadIdx.x + offset + primitiveInfo.x]);
            }

            if (threadIdx.x == 0){
                minResult[segmentID] = min(aabbMin[primitiveInfo.x], aabbMin[1 + primitiveInfo.x]);
                maxResult[segmentID] = max(aabbMax[primitiveInfo.x], aabbMax[1 + primitiveInfo.x]);
            }
        }
    }


    /*
    __global__ void SegmentedReduce0(float4 *segmentAabbMin,
                                    float4 *segmentAabbMax,
                                    int *segmentOwner,
                                    float4 *nodeAabbMin,
                                    float4 *nodeAabbMax){

        const unsigned int segmentID = threadIdx.x;
        int cnt = threadIdx.x;
        while (cnt < d_activeNodeRange){
            nodeAabbMin[cnt + d_activeNodeIndex] = make_float4(fInfinity);
            nodeAabbMax[cnt + d_activeNodeIndex] = make_float4(-1.0 * fInfinity);
            cnt += blockDim.x;
        }
        __syncthreads();

        int index0 = segmentID*2;
        int index1 = index0 + 1;
        int offset = blockDim.x;

        while (index1 < d_segments){
            int i0 = index0;
            int i1 = index1;
            while (i1 < d_segments){
                int owner0 = segmentOwner[i0];
                int owner1 = segmentOwner[i1];
                
                if (owner0 != owner1){
                    nodeAabbMin[owner1] = min(nodeAabbMin[owner1], segmentAabbMin[i1]);
                    nodeAabbMax[owner1] = max(nodeAabbMax[owner1], segmentAabbMax[i1]);
                }else{
                    segmentAabbMin[i0] = min(segmentAabbMin[i0], segmentAabbMin[i1]);
                    segmentAabbMax[i0] = max(segmentAabbMax[i0], segmentAabbMax[i1]);
                }

                i0 += offset;
                i1 += offset;
            }

            index0 *= 2;
            index1 *= 2;
            offset *= 2;
            __syncthreads();
        }

        if (threadIdx.x == 0){
            nodeAabbMin[segmentOwner[0]] = segmentAabbMin[0];
            nodeAabbMax[segmentOwner[0]] = segmentAabbMax[0];
        }
    }
    */

    __global__ void SegmentedReduce0(float4 *segmentAabbMin,
                                     float4 *segmentAabbMax,
                                     int *segmentOwner,
                                     float4 *nodeAabbMin,
                                     float4 *nodeAabbMax){

        const unsigned int segmentID = threadIdx.x;
        if (threadIdx.x < d_activeNodeRange){
            nodeAabbMin[threadIdx.x + d_activeNodeIndex] = make_float4(fInfinity);
            nodeAabbMax[threadIdx.x + d_activeNodeIndex] = make_float4(-1.0 * fInfinity);
        }
        __syncthreads();

        int index0 = segmentID * 2;
        int index1 = index0 + 1;

        while (index1 < d_segments){
            int owner0 = segmentOwner[index0];
            int owner1 = segmentOwner[index1];
            
            if (owner0 != owner1){
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
            nodeAabbMin[segmentOwner[0]] = segmentAabbMin[0];
            nodeAabbMax[segmentOwner[0]] = segmentAabbMax[0];
        }
    }

    /*
    __global__ void SegmentedReduce1(float4 *segmentAabbMin,
                                     float4 *segmentAabbMax,
                                     int *segmentOwner,
                                     float4 *nodeAabbMin,
                                     float4 *nodeAabbMax){

        __shared__ float3 segMax[256];
        __shared__ float4 resMax[256];
        
        segMax[threadIdx.x] = threadIdx.x < d_segments ? make_float3(segmentAabbMax[threadIdx.x]) : make_float3(-1.0f * fInfinity);
        if (threadIdx.x < d_activeNodeRange){
            nodeAabbMin[threadIdx.x] = make_float4(fInfinity);
            resMax[threadIdx.x] = nodeAabbMax[threadIdx.x] = make_float4(-1.0 * fInfinity);
        }
        __syncthreads();

        int index0 = threadIdx.x * 2;
        int index1 = index0 + 1;
        while (index1 < d_segments){
            int owner0 = segmentOwner[index0] - d_activeNodeIndex;
            int owner1 = segmentOwner[index1] - d_activeNodeIndex;
            
            if (owner0 != owner1){
                nodeAabbMin[owner1] = min(nodeAabbMin[owner1], segmentAabbMin[index1]);
                resMax[owner1] = nodeAabbMax[owner1] = max(resMax[owner1], make_float4(segMax[index1], 1.0f));
                //resMax[owner1] = max(resMax[owner1], make_float4(segMax[index1], 1.0f));
                //nodeAabbMax[owner1] = make_float4(threadIdx.x);
            }else{
                segmentAabbMin[index0] = min(segmentAabbMin[index0], segmentAabbMin[index1]);
                segMax[index0] = max(segMax[index0], segMax[index1]);
            }
 
            __syncthreads();
           index0 *= 2;
           index1 *= 2;
        }
        __syncthreads();

        if (threadIdx.x < d_activeNodeRange)
            nodeAabbMax[threadIdx.x] = resMax[threadIdx.x];

        if (threadIdx.x == 0){
            nodeAabbMin[0] = segmentAabbMin[0];
            nodeAabbMax[0] = make_float4(segMax[0], 1.0f);
        }
    }
    */

    /*
    __global__ void 
    __launch_bounds__(256) 
        FinalSegmentedReduce(float4 *segmentAabbMin,
                             float4 *segmentAabbMax,
                             int *segmentOwner,
                             float4 *resAabbMin,
                             float4 *resAabbMax){
        
        __shared__ float3 segMin[256];
        __shared__ float3 segMax[256];
        __shared__ int segOwner[256];
        __shared__ float3 resMin[256];
        __shared__ float3 resMax[256];

        segMin[threadIdx.x] = threadIdx.x < d_segments ? make_float3(segmentAabbMin[threadIdx.x]) : make_float3(fInfinity);
        segMax[threadIdx.x] = threadIdx.x < d_segments ? make_float3(segmentAabbMax[threadIdx.x]) : make_float3(-1.0f * fInfinity);
        segOwner[threadIdx.x] = threadIdx.x < d_segments ? segmentOwner[threadIdx.x] : 0;
        resMin[threadIdx.x] = make_float3(fInfinity);
        resMax[threadIdx.x] = make_float3(-1.0f * fInfinity);

        __syncthreads();

        int index0 = threadIdx.x * 2;
        int index1 = index0 + 1;

        while (index1 < d_segments){
            int owner0 = segOwner[index0] - d_activeNodeIndex;
            int owner1 = segOwner[index1] - d_activeNodeIndex;

            if (owner0 != owner1){
                resMin[owner1] = min(resMin[owner1], segMin[index1]);
                resMax[owner1] = max(resMax[owner1], segMax[index1]);
            }else{
                segMin[index0] = min(segMin[index0], segMin[index1]);
                segMax[index0] = max(segMax[index0], segMax[index1]);
            }

            index0 *= 2;
            index1 *= 2;
            __syncthreads();
        }

        if (threadIdx.x == 0){
            resMin[0] = segMin[0];
            resMax[0] = segMax[0];
        }

        if (threadIdx.x < d_activeNodeRange){
            resAabbMin[threadIdx.x] = make_float4(resMin[threadIdx.x], 1.0f);
            resAabbMax[threadIdx.x] = make_float4(resMax[threadIdx.x], 1.0f);
        }
    }
*/
    
    /*    
    __global__ void SegmentedReduce1(float4 *segmentAabbMin,
                                     float4 *segmentAabbMax,
                                     int *segmentOwner,
                                     float4 *nodeAabbMin,
                                     float4 *nodeAabbMax){
        __shared__ float3 segMin[256];
        __shared__ float3 segMax[256];
        __shared__ float3 resMin[256];
        __shared__ float3 resMax[256];
        __shared__ int owner[256];

        int localOwner = segmentOwner[0];
        float3 localMin = make_float3(segmentAabbMin[0]);
        float3 localMax = make_float3(segmentAabbMax[0]);

        for (int offset = 0; offset < d_segments; offset += 256){
            if (threadIdx.x + offset < d_segments){
                segMin[threadIdx.x] = make_float3(segmentAabbMin[threadIdx.x + offset]);
                segMax[threadIdx.x] = make_float3(segmentAabbMax[threadIdx.x + offset]);
                owner[threadIdx.x] = segmentOwner[threadIdx.x + offset];
                resMin[threadIdx.x] = make_float3(fInfinity);
                resMax[threadIdx.x] = make_float3(-1.0f * fInfinity);
            }
            __syncthreads();

            int index0 = threadIdx.x + offset;
            int index1 = index0+1;
        }
    }
    */

    __global__ void CalcUpperNodeSplitInfo(float4 *aabbMin, float4* aabbMax,
                                          float *splitPos, char *info){
        const int id = blockDim.x * blockIdx.x + threadIdx.x;
        
        if (id < d_activeNodeRange){
            
            float4 bbSize = aabbMax[id] - aabbMin[id];
            float4 median = bbSize * 0.5 + aabbMin[id];

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

    
}
}
}
}
