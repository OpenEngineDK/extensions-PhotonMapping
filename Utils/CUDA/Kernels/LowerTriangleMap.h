#include <Meta/CUDA.h>
#include <Utils/CUDA/Kernels/PhotonMapDeviceVars.h>
#include <Utils/CUDA/SharedMemory.h>
#include <Scene/TriangleNode.h>

namespace OpenEngine {
    using namespace Scene;
namespace Utils {
namespace CUDA {
namespace Kernels {

    __global__ void PreprocesLowerNodes(int *upperLeafIDs,
                                        char* upperNodeInfo,
                                        int2 *primitiveInfo,
                                        float* surfaceArea,
                                        float4* primMax,
                                        int *upperLeft, int *upperRight,
                                        int activeIndex, int activeRange){

        const int id = blockDim.x * blockIdx.x + threadIdx.x;
                
        if (id < activeRange){
            int leafID = upperLeafIDs[id];
            int lowerNodeID = id + activeIndex;
            upperNodeInfo[leafID] = KDNode::PROXY;
            int2 triInfo = primitiveInfo[leafID];

            float area = 0.0f;
            for (int i = 0; i < triInfo.y; ++i)
                area += primMax[triInfo.x + i].w;
            surfaceArea[leafID] = area;

            triInfo.y = (1<<triInfo.y)-1;
            primitiveInfo[lowerNodeID] = triInfo;
            upperLeft[leafID] = upperRight[leafID] = lowerNodeID;

        }
    }

    __global__ void CreateSplittingPlanes(int4 *splitTriangleSet,
                                          int2 *primitiveInfo,
                                          float4* aabbMins, float4* aabbMaxs,
                                          int activeRange){

        const int id = blockDim.x * blockIdx.x + threadIdx.x;
        const int nodeID = id / TriangleNode::MAX_LOWER_SIZE;

        if (nodeID < activeRange){
            const int2 triInfo = primitiveInfo[nodeID];
            const char tris = bitcount(triInfo.y);
            const int triID = id % TriangleNode::MAX_LOWER_SIZE;
            const int triIndex = triInfo.x + triID;

            // Copy aabbs to shared mem.
            float3* aabbMin = SharedMemory<float3>();
            float4* aabbMax = SharedMemory<float4>();
            const float3 lowSplitPlane = aabbMin[threadIdx.x] = 
                triID < tris ? make_float3(aabbMins[triIndex]) : make_float3(0.0f);
            aabbMax[threadIdx.x] = triID < tris ? aabbMaxs[triIndex] : make_float4(0.0f);
            const float3 highSplitPlane = make_float3(aabbMax[threadIdx.x]);

            // Is automatically optimized away by the compiler. nvcc
            // actually works sometimes.
            if (TriangleNode::MAX_LOWER_SIZE > warpSize)
                __syncthreads();

            int4 splitX = make_int4(0, 0, 0, 0); // {lowLeft, lowRight, highLeft, highRight}
            int4 splitY = make_int4(0, 0, 0, 0); int4 splitZ = make_int4(0, 0, 0, 0);

            //#pragma unroll
            //for (char i = 0; i < TriangleNode::MAX_LOWER_SIZE; ++i){
            for (char i = 0; i < tris; ++i){
                int index = nodeID * TriangleNode::MAX_LOWER_SIZE + i;

                float3 minCorner = aabbMin[index];
                float4 maxCorner = aabbMax[index];

                int activePrim = maxCorner.w > 0.0f ? 1 : 0;

                splitX.x += minCorner.x < lowSplitPlane.x ? activePrim<<i : 0;
                splitX.y += lowSplitPlane.x < maxCorner.x ? activePrim<<i : 0;
                splitX.z += minCorner.x < highSplitPlane.x ? activePrim<<i : 0;
                splitX.w += highSplitPlane.x < maxCorner.x ? activePrim<<i : 0;

                splitY.x += minCorner.y < lowSplitPlane.y ? activePrim<<i : 0;
                splitY.y += lowSplitPlane.y < maxCorner.y ? activePrim<<i : 0;
                splitY.z += minCorner.y < highSplitPlane.y ? activePrim<<i : 0;
                splitY.w += highSplitPlane.y < maxCorner.y ? activePrim<<i : 0;

                splitZ.x += minCorner.z < lowSplitPlane.z ? activePrim<<i : 0;
                splitZ.y += lowSplitPlane.z < maxCorner.z ? activePrim<<i : 0;
                splitZ.z += minCorner.z < highSplitPlane.z ? activePrim<<i : 0;
                splitZ.w += highSplitPlane.z < maxCorner.z ? activePrim<<i : 0;
            }

            // @OPT Left split set should be smallest to facilitate
            // better thread coherence.

            if (triID < tris){
                splitTriangleSet[triIndex] = splitX;
                splitTriangleSet[d_triangles + triIndex] = splitY;
                splitTriangleSet[2 * d_triangles + triIndex] = splitZ;
            }
        }
    }

    __host__ void CalcSAHForSets(int4 splittingSets, int areaIndices, float* areas, 
                                 float &optimalSAH, 
                                 float &leftArea, float &rightArea,
                                 int &leftSet, int &rightSet){
        
        float4 setAreas = make_float4(0.0f);
        
        while (areaIndices){
            int i = ffs(areaIndices) - 1;
            
            setAreas.x += splittingSets.x & (1<<i) ? areas[i] : 0.0f;
            setAreas.y += splittingSets.y & (1<<i) ? areas[i] : 0.0f;
            setAreas.z += splittingSets.z & (1<<i) ? areas[i] : 0.0f;
            setAreas.w += splittingSets.w & (1<<i) ? areas[i] : 0.0f;

            areaIndices -= 1<<i;
        }
        
        float lowSAH = bitcount(splittingSets.x) * setAreas.x + bitcount(splittingSets.y) * setAreas.y;
        float highSAH = bitcount(splittingSets.z) * setAreas.z + bitcount(splittingSets.w) * setAreas.w;

        if (lowSAH > highSAH){
            leftSet = splittingSets.x;
            rightSet = splittingSets.y;
            leftArea = setAreas.x;
            rightArea = setAreas.y;
            optimalSAH = lowSAH;
        }else{
            leftSet = splittingSets.z;
            rightSet = splittingSets.w;
            leftArea = setAreas.z;
            rightArea = setAreas.w;
            optimalSAH = highSAH;
        }
    }
    
    // @OPT move surfacearea to a single float array?
    __global__ void CalcSAH(int2 *primitiveInfo,
                            float4 *aabbMax,
                            int4 *splitTriangleSet,
                            int* indices){
        const int id = blockDim.x * blockIdx.x + threadIdx.x;
        
        if (id < d_activeNodeRange){
            const int2 primInfo = primitiveInfo[id];
            
            // @OPT. Perhaps the threads can fill the area array coalesced?
            float* area = SharedMemory<float>();

            int bitmap = primInfo.y;
            while(bitmap){
                int index = __ffs(bitmap) - 1;
                area[index + TriangleNode::MAX_LOWER_SIZE * threadIdx.x] = aabbMax[primInfo.x + index].w;
                bitmap -= 1<<index;
            }

            float optimalSAH = fInfinity;
            int optimalPlane;

            int triangles = primInfo.y;
            while(triangles){
                int i = __ffs(triangles) - 1;

                int4 splittingSet = splitTriangleSet[primInfo.x + i];
                float4 SAH = make_float4(0.0f);

                int surfaces = primInfo.y;
                while (surfaces){
                    int j = __ffs(surfaces) - 1;

                    SAH.x += splittingSet.x & (1<<j) ? area[j] : 0.0f;
                    SAH.y += splittingSet.y & (1<<j) ? area[j] : 0.0f;
                    SAH.z += splittingSet.z & (1<<j) ? area[j] : 0.0f;
                    SAH.w += splittingSet.w & (1<<j) ? area[j] : 0.0f;

                    surfaces -= 1<<j;
                }

                // lower x split
                SAH.x = __popc(primInfo.y & splittingSet.x) * SAH.x + __popc(primInfo.y & splittingSet.y) * SAH.y;
                if (SAH.x < optimalSAH){
                    optimalSAH = SAH.x; optimalPlane = primInfo.x + i;
                }
                // higher x split
                SAH.y = __popc(primInfo.y & splittingSet.z) * SAH.z + __popc(primInfo.y & splittingSet.w) * SAH.w;
                if (SAH.y < optimalSAH){
                    optimalSAH = SAH.y; optimalPlane = primInfo.x + i + 1<<31;
                }

                triangles -= 1<<i;
            }
            
        }
    }

}
}
}
}
