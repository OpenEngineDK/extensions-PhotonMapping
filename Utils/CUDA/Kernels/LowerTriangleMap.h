#include <Meta/CUDA.h>
#include <Utils/CUDA/Kernels/PhotonMapDeviceVars.h>
#include <Utils/CUDA/SharedMemory.h>
#include <Scene/TriangleNode.h>

namespace OpenEngine {
    using namespace Scene;
namespace Utils {
namespace CUDA {
namespace Kernels {

#define traverselCost 2.0f

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
            int2 triInfo = primitiveInfo[leafID];

            float area = 0.0f;
            int size = triInfo.y;
            triInfo.y = 0;
            for (int i = 0; i < size; ++i){
                float a = primMax[triInfo.x + i].w;
                triInfo.y += a > 0.0f ? (1<<i) : 0;
                area += a;
            }
            surfaceArea[lowerNodeID] = area;

            upperNodeInfo[leafID] = area > 0.0f ? KDNode::PROXY : KDNode::LEAF;

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
            const int2 primInfo = primitiveInfo[nodeID];
            const int primID = id % TriangleNode::MAX_LOWER_SIZE;
            const int primIndex = primInfo.x + primID;

            // Copy aabbs to shared mem.
            float3* aabbMin = SharedMemory<float3>();
            float3* aabbMax = aabbMin + blockDim.x;

            const float3 lowSplitPlane = aabbMin[threadIdx.x] = 
                primInfo.y & 1<<primID ? make_float3(aabbMins[primIndex]) : make_float3(0.0f);
            const float3 highSplitPlane = aabbMax[threadIdx.x] = 
                primInfo.y & 1<<primID ? make_float3(aabbMaxs[primIndex]) : make_float3(0.0f);

            // Is automatically optimized away by the compiler. nvcc
            // actually works sometimes.
            if (TriangleNode::MAX_LOWER_SIZE > warpSize)
                __syncthreads();

            int4 splitX = make_int4(0, 0, 0, 0); // {lowLeft, lowRight, highLeft, highRight}
            int4 splitY = make_int4(0, 0, 0, 0); int4 splitZ = make_int4(0, 0, 0, 0);

            int sharedOffset = threadIdx.x - primID;

            int triangles = primInfo.y;
            while(triangles){
                int i = __ffs(triangles) - 1;

                float3 minCorner = aabbMin[sharedOffset + i];
                float3 maxCorner = aabbMax[sharedOffset + i];

                splitX.x |= minCorner.x <= lowSplitPlane.x ? 1<<i : 0;
                splitX.y |= lowSplitPlane.x < maxCorner.x ? 1<<i : 0;
                splitX.z |= minCorner.x <= highSplitPlane.x ? 1<<i : 0;
                splitX.w |= highSplitPlane.x < maxCorner.x ? 1<<i : 0;

                splitY.x |= minCorner.y <= lowSplitPlane.y ? 1<<i : 0;
                splitY.y |= lowSplitPlane.y < maxCorner.y ? 1<<i : 0;
                splitY.z |= minCorner.y <= highSplitPlane.y ? 1<<i : 0;
                splitY.w |= highSplitPlane.y < maxCorner.y ? 1<<i : 0;

                splitZ.x |= minCorner.z <= lowSplitPlane.z ? 1<<i : 0;
                splitZ.y |= lowSplitPlane.z < maxCorner.z ? 1<<i : 0;
                splitZ.z |= minCorner.z <= highSplitPlane.z ? 1<<i : 0;
                splitZ.w |= highSplitPlane.z < maxCorner.z ? 1<<i : 0;
                
                triangles -= 1<<i;
            }

            // @OPT Left split set should be smallest to facilitate
            // better thread coherence.

            if (primInfo.y & 1<<primID){
                splitTriangleSet[primIndex] = splitX;
                splitTriangleSet[d_triangles + primIndex] = splitY;
                splitTriangleSet[2 * d_triangles + primIndex] = splitZ;
            }
            
        }
    }
    
    __device__ void CalcAreaForSets(int4 splittingSets, char splitAxis, 
                                    int setIndex,
                                    int areaIndices, float* areas, 
                                    float &optimalArea, 
                                    float &leftArea, float &rightArea,
                                    int &leftSet, int &rightSet,
                                    char &optimalAxis,
                                    int &splitIndex){
        
        float4 setAreas = make_float4(0.0f);
        
        splittingSets.x &= areaIndices;
        splittingSets.y &= areaIndices;
        splittingSets.z &= areaIndices;
        splittingSets.w &= areaIndices;

        while (areaIndices){
            int i = __ffs(areaIndices) - 1;
            
            setAreas.x += splittingSets.x & (1<<i) ? areas[i] : 0.0f;
            setAreas.y += splittingSets.y & (1<<i) ? areas[i] : 0.0f;
            setAreas.z += splittingSets.z & (1<<i) ? areas[i] : 0.0f;
            setAreas.w += splittingSets.w & (1<<i) ? areas[i] : 0.0f;

            areaIndices -= 1<<i;
        }
        
        float lowArea = __popc(splittingSets.x) * setAreas.x + __popc(splittingSets.y) * setAreas.y;
        float highArea = __popc(splittingSets.z) * setAreas.z + __popc(splittingSets.w) * setAreas.w;

        if (lowArea < optimalArea){
            leftSet = splittingSets.x;
            rightSet = splittingSets.y;
            leftArea = setAreas.x;
            rightArea = setAreas.y;
            optimalArea = lowArea;
            optimalAxis = splitAxis;
            splitIndex = setIndex;
        }

        if (highArea < optimalArea){
            leftSet = splittingSets.z;
            rightSet = splittingSets.w;
            leftArea = setAreas.z;
            rightArea = setAreas.w;
            optimalArea = highArea;
            optimalAxis = splitAxis;
            splitIndex = setIndex | (1<<31);
        }
    }

    // @OPT move surfacearea to a single float array?
    __global__ void 
    __launch_bounds__(96) 
        CalcSAH(char *info,
            float *splitPoss,
            int2 *primitiveInfo,
            float *nodeSurface,
            float4 *aabbMin, float4 *aabbMax,
            int4 *splitTriangleSet,
            float2 *childAreas,
            int2 *childSets,
            int *splitSides){
        const int id = blockDim.x * blockIdx.x + threadIdx.x;
        
        if (id < d_activeNodeRange){
            const int2 primInfo = primitiveInfo[id];
            
            // @OPT. Perhaps the threads can fill the area array coalesced?
            float* area = SharedMemory<float>();
            area += TriangleNode::MAX_LOWER_SIZE * threadIdx.x;
            //float area[32];

            int bitmap = primInfo.y;
            while(bitmap){
                int index = __ffs(bitmap) - 1;
                area[index] = aabbMax[primInfo.x + index].w;
                bitmap -= 1<<index;
            }            

            // @OPT. Calculate the area for bitmap pairs? Then areas
            // can be summed over 2 bits at a time. Or template to N
            // bits. For fun and profit.

            float optimalArea = fInfinity;
            float leftArea, rightArea;
            int leftSet, rightSet;
            char axis;
            int splitIndex;

            int triangles = primInfo.y;
            while(triangles){
                int i = __ffs(triangles) - 1;

                CalcAreaForSets(splitTriangleSet[primInfo.x + i], KDNode::X,
                                primInfo.x + i,
                                primInfo.y, area, 
                                optimalArea, 
                                leftArea, rightArea,
                                leftSet, rightSet, axis, splitIndex);

                CalcAreaForSets(splitTriangleSet[d_triangles + primInfo.x + i], KDNode::Y,
                                primInfo.x + i,
                                primInfo.y, area, 
                                optimalArea, 
                                leftArea, rightArea,
                                leftSet, rightSet, axis, splitIndex);

                CalcAreaForSets(splitTriangleSet[2 * d_triangles + primInfo.x + i], KDNode::Z,
                                primInfo.x + i,
                                primInfo.y, area, 
                                optimalArea, 
                                leftArea, rightArea,
                                leftSet, rightSet, axis, splitIndex);

                triangles -= 1<<i;
            }
            
            float nodeArea = nodeSurface[id];
            bool split = optimalArea < (__popc(primInfo.y) - traverselCost) * nodeArea;
            if (split){
                // Dump stuff and move on.
                childAreas[id] = make_float2(leftArea, rightArea);
                childSets[id] = make_int2(leftSet, rightSet);
                float3 splitPositions;
                if (splitIndex & 1<<31){
                    // A high splitplane was used
                    splitPositions = make_float3(aabbMax[splitIndex ^ 1<<31]);
                }else{
                    // A low splitplane was used
                    splitPositions = make_float3(aabbMin[splitIndex]);
                }
                splitPoss[id] = axis == KDNode::X ? splitPositions.x : (axis == KDNode::Y ? splitPositions.y : splitPositions.z);
            }
            info[id] = split ? axis : KDNode::LEAF;
            splitSides[id] = split;
        }
    }

    __global__ void CreateLowerSAHChildren(int *childSplit,
                                           int *childAddrs,
                                           float2 *childAreas,
                                           int2 *childSets,
                                           float* nodeArea,
                                           int2* primitiveInfo,
                                           int nodeSplits){

        // @OPT 'or' the childSets onto float4 nodeArea. That way we
        // can get everything in one store/lookup?
        
        const int id = blockDim.x * blockIdx.x + threadIdx.x;
        
        if (id < d_activeNodeRange){
            int split = childSplit[id];

            if (split){
                float2 childrenArea = childAreas[id];
                int2 childrenSet = childSets[id];
                
                const int childOffset = childAddrs[id];
                
                const int parentID = d_activeNodeIndex + id;
                int2 parentPrimInfo = primitiveInfo[parentID];
                
                const int leftChildID = d_activeNodeIndex + d_activeNodeRange + childOffset;
                nodeArea[leftChildID] = childrenArea.x;
                primitiveInfo[leftChildID] = make_int2(parentPrimInfo.x, childrenSet.x);
                
                const int rightChildID = leftChildID + nodeSplits;
                nodeArea[rightChildID] = childrenArea.y;
                primitiveInfo[rightChildID] = make_int2(parentPrimInfo.x, childrenSet.y);
            }
        }        
    }

}
}
}
}
