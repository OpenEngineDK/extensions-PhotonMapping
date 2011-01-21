#ifndef _CUDA_KERNEL_LOWER_TRIANGLE_MAP_H_
#define _CUDA_KERNEL_LOWER_TRIANGLE_MAP_H_

#include <Utils/CUDA/SharedMemory.h>
#include <Utils/CUDA/Utils.h>

/*
 * Dependencies
#include <Scene/TriangleNode.h>
#include <Utils/CUDA/Kernels/PhotonMapDeviceVars.h>

using namespace OpenEngine::Scene;
using namespace OpenEngine::Utils::CUDA::Kernels;
*/

#define traverselCost 32.0f
#define minLeafTriangles 32

// @TODO use the reduced bounding box and actual bounding box'es
// diagonals to estimate the sub triangles surface area.  Or the
// amount of aabb reduction in each dimension multiplied? How will
// that handle 0 sized sides?

__global__ void CalcSurfaceArea(int *indices, 
                                float4 *v0s, float4 *v1s, float4 *v2s,
                                float *areas,
                                int triangles){

    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (id < triangles){
        int index = indices[id];
        const float3 v0 = make_float3(v0s[index]);
        const float3 v1 = make_float3(v1s[index]);
        const float3 v2 = make_float3(v2s[index]);
        areas[id] = 0.5f * length(cross(v1-v0, v2-v0));
    }
}

__global__ void PreprocessLeafNodes(int *upperLeafIDs,
                                    KDNode::bitmap *primBitmap,
                                    int activeRange){

    const int id = blockDim.x * blockIdx.x + threadIdx.x;
                
    if (id < activeRange){
        int leafID = upperLeafIDs[id];
        
        KDNode::bitmap bmp = primBitmap[leafID];
        bmp = (KDNode::bitmap(1)<<bmp)-1;
        
        primBitmap[leafID] = bmp;
    }
}

__global__ void PreprocesLowerNodes(int *upperLeafIDs,
                                    int *primIndices,
                                    KDNode::bitmap *primBitmap,
                                    float* surfaceArea,
                                    float* primAreas,
                                    int activeRange){

    const int id = blockDim.x * blockIdx.x + threadIdx.x;
                
    if (id < activeRange){
        int leafID = upperLeafIDs[id];
        int primIndex = primIndices[leafID];
        KDNode::amount primAmount = primBitmap[leafID];

        float area = 0.0f;
        KDNode::bitmap bmp = 0;
        for (int i = 0; i < primAmount; ++i){
            float a = primAreas[primIndex + i];
            bmp += a > 0.0f ? (KDNode::bitmap(1)<<i) : 0;
            area += a;
        }
        
        surfaceArea[leafID] = area;
        primBitmap[leafID] = bmp;
    }
}

__global__ void CreateSplittingPlanes(int *upperLeafIDs,
                                      int *primitiveIndex, KDNode::bitmap *primBitmap,
                                      float4* aabbMins, float4* aabbMaxs,
                                      KDNode::bitmap4 *splitTriangleSet,
                                      int activeRange){

    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    int nodeID = id / KDNode::MAX_LOWER_SIZE;

    if (nodeID < activeRange){
    
        nodeID = upperLeafIDs[nodeID];
        
        const int primID = id % KDNode::MAX_LOWER_SIZE;
        const int primIndex = primitiveIndex[nodeID] + primID;
        const KDNode::bitmap primBmp = primBitmap[nodeID];

        // Copy aabbs to shared mem.
        float3* aabbMin = SharedMemory<float3>();
        float3* aabbMax = aabbMin + blockDim.x;

        const float3 lowSplitPlane = aabbMin[threadIdx.x] = 
            primBmp & KDNode::bitmap(1)<<primID ? make_float3(aabbMins[primIndex]) : make_float3(0.0f);
        const float3 highSplitPlane = aabbMax[threadIdx.x] = 
            primBmp & KDNode::bitmap(1)<<primID ? make_float3(aabbMaxs[primIndex]) : make_float3(0.0f);

        // Is automatically optimized away by the compiler. nvcc
        // actually works sometimes.
        if (KDNode::MAX_LOWER_SIZE > warpSize)
            __syncthreads();

        KDNode::bitmap4 splitX = KDNode::make_bitmap4(0, 0, 0, 0); // {lowLeft, lowRight, highLeft, highRight}
        KDNode::bitmap4 splitY = KDNode::make_bitmap4(0, 0, 0, 0); KDNode::bitmap4 splitZ = KDNode::make_bitmap4(0, 0, 0, 0);

        int sharedOffset = threadIdx.x - primID;

        KDNode::bitmap triangles = primBmp;
        while(triangles){
            int i = firstBitSet(triangles) - 1;

            float3 minCorner = aabbMin[sharedOffset + i];
            float3 maxCorner = aabbMax[sharedOffset + i];

            splitX.x |= minCorner.x <= lowSplitPlane.x ? KDNode::bitmap(1)<<i : 0;
            splitX.y |= lowSplitPlane.x < maxCorner.x ? KDNode::bitmap(1)<<i : 0;
            splitX.z |= minCorner.x <= highSplitPlane.x ? KDNode::bitmap(1)<<i : 0;
            splitX.w |= highSplitPlane.x < maxCorner.x ? KDNode::bitmap(1)<<i : 0;

            splitY.x |= minCorner.y <= lowSplitPlane.y ? KDNode::bitmap(1)<<i : 0;
            splitY.y |= lowSplitPlane.y < maxCorner.y ? KDNode::bitmap(1)<<i : 0;
            splitY.z |= minCorner.y <= highSplitPlane.y ? KDNode::bitmap(1)<<i : 0;
            splitY.w |= highSplitPlane.y < maxCorner.y ? KDNode::bitmap(1)<<i : 0;

            splitZ.x |= minCorner.z <= lowSplitPlane.z ? KDNode::bitmap(1)<<i : 0;
            splitZ.y |= lowSplitPlane.z < maxCorner.z ? KDNode::bitmap(1)<<i : 0;
            splitZ.z |= minCorner.z <= highSplitPlane.z ? KDNode::bitmap(1)<<i : 0;
            splitZ.w |= highSplitPlane.z < maxCorner.z ? KDNode::bitmap(1)<<i : 0;
                
            triangles -= KDNode::bitmap(1)<<i;
        }

        if (primBmp & 1<<primID){
            splitTriangleSet[primIndex] = splitX;
            splitTriangleSet[d_triangles + primIndex] = splitY;
            splitTriangleSet[2 * d_triangles + primIndex] = splitZ;
        }
            
    }
}

__device__ __host__ void CalcRelationForSets(KDNode::bitmap4 splittingSet, KDNode::bitmap nodeSet,
                                             char splitAxis, int setIndex, 
                                             float &optimalRelation,
                                             int &largestSize,
                                             KDNode::bitmap &leftSet, KDNode::bitmap &rightSet,
                                             char &optimalAxis,
                                             int &splitIndex){
    
    splittingSet.x &= nodeSet;
    splittingSet.y &= nodeSet;
    splittingSet.z &= nodeSet;
    splittingSet.w &= nodeSet;

    int small = min(bitcount(splittingSet.x), bitcount(splittingSet.y));
    int large = max(bitcount(splittingSet.x), bitcount(splittingSet.y));
    float rel = small / float(large);
    
    if (large < largestSize || (large == largestSize && rel < optimalRelation)){
        optimalRelation = rel;
        largestSize = large;
        leftSet = splittingSet.x; rightSet = splittingSet.y;
        optimalAxis = splitAxis;
        splitIndex = setIndex;
    }

    small = min(bitcount(splittingSet.z), bitcount(splittingSet.w));
    large = max(bitcount(splittingSet.z), bitcount(splittingSet.w));
    rel = small / float(large);
    
    if (large < largestSize || (large == largestSize && rel < optimalRelation)){
        optimalRelation = rel;
        largestSize = large;
        leftSet = splittingSet.z; rightSet = splittingSet.w;
        optimalAxis = splitAxis;
        splitIndex = setIndex | (1<<31);
    }
}

// @OPT. Splitting the axis and splitting sets into float and int2
// arrays in a preprocess, might make the kernels faster, since we can
// use index lookups to avoid branching and store fewer values

template <bool useIndices>
__global__ void 
CalcSplit(int *upperLeafIDs,
          char *info,
          float *splitPoss,
          int *primitiveIndex, KDNode::bitmap *primitiveBitmap,
          float4 *aabbMin, float4 *aabbMax,
          KDNode::bitmap4 *splitTriangleSet,
          KDNode::bitmap2 *childSets,
          int *splitSides){
    
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
        
    if (id < d_activeNodeRange){
        const int parentID = useIndices ? upperLeafIDs[id] : d_activeNodeIndex + id;

        const int primIndex = primitiveIndex[parentID];
        const KDNode::bitmap primBmp = primitiveBitmap[parentID];

        float relation = 0.0f;
        int largestSetSize = TriangleNode::MAX_LOWER_SIZE;
        KDNode::bitmap leftSet, rightSet;
        char axis;
        int splitIndex;
        
        int triangles = primBmp;
        while(triangles){
            int i = firstBitSet(triangles) - 1;
            
            CalcRelationForSets(splitTriangleSet[primIndex + i], primBmp,
                                KDNode::X, primIndex + i,
                                relation, largestSetSize,
                                leftSet, rightSet,
                                axis, splitIndex);
            
            CalcRelationForSets(splitTriangleSet[d_triangles + primIndex + i], primBmp,
                                KDNode::Y, primIndex + i,
                                relation, largestSetSize,
                                leftSet, rightSet,
                                axis, splitIndex);
            
            CalcRelationForSets(splitTriangleSet[2 * d_triangles + primIndex + i], primBmp,
                                KDNode::Z, primIndex + i,
                                relation, largestSetSize,
                                leftSet, rightSet,
                                axis, splitIndex);
            
            triangles -= KDNode::bitmap(1)<<i;
        }

        // Weird stuff, how about stopping if largestSetSize ==
        // nodeSize or nodeSize < minLeafTriangles?

        bool split = minLeafTriangles < largestSetSize * relation;
        if (split){
            // Dump stuff and move on
            childSets[id] = KDNode::make_bitmap2(leftSet, rightSet);
            float3 splitPositions;
            if (splitIndex & 1<<31){
                // A high splitplane was used
                splitPositions = make_float3(aabbMax[splitIndex ^ 1<<31]);
            }else{
                // A low splitplane was used
                splitPositions = make_float3(aabbMin[splitIndex]);
            }
            splitPoss[parentID] = axis == KDNode::X ? splitPositions.x : (axis == KDNode::Y ? splitPositions.y : splitPositions.z);
        }
        info[parentID] = split ? axis : KDNode::LEAF;
        splitSides[id] = split;
    }
}

template <bool useIndices>
__global__ void CreateChildren(int *upperLeafIDs,
                               int *childSplit,
                               int *childAddrs,
                               KDNode::bitmap2 *childSets,
                               int* primitiveIndex,
                               KDNode::bitmap* primitiveBitmap,
                               int2 *children, 
                               int nodeSplits){

    const int id = blockDim.x * blockIdx.x + threadIdx.x;
        
    if (id < d_activeNodeRange){
        int split = childSplit[id];

        if (split){
            KDNode::bitmap2 childrenSet = childSets[id];
                
            const int childOffset = childAddrs[id];

            const int parentID = useIndices ? upperLeafIDs[id] : d_activeNodeIndex + id;
            int parentPrimIndex = primitiveIndex[parentID];
                
            const int leftChildID = useIndices 
                ? d_activeNodeIndex + childOffset 
                : d_activeNodeIndex + d_activeNodeRange + childOffset;
            primitiveIndex[leftChildID] = parentPrimIndex;
            primitiveBitmap[leftChildID] = childrenSet.x;
                
            const int rightChildID = leftChildID + nodeSplits;
            primitiveIndex[rightChildID] = parentPrimIndex;
            primitiveBitmap[rightChildID] = childrenSet.y;

            children[parentID] = make_int2(leftChildID, rightChildID);
        }
    }        
}


// **** SURFACE AREA HEURISTIC *****

__device__ void CalcAreaForSets(KDNode::bitmap4 splittingSets, char splitAxis, 
                                int setIndex,
                                KDNode::bitmap areaIndices, float* areas, 
                                float &optimalArea, 
                                float &leftArea, float &rightArea,
                                KDNode::bitmap &leftSet, KDNode::bitmap &rightSet,
                                char &optimalAxis,
                                int &splitIndex){
        
    float4 setAreas = make_float4(0.0f);
        
    splittingSets.x &= areaIndices;
    splittingSets.y &= areaIndices;
    splittingSets.z &= areaIndices;
    splittingSets.w &= areaIndices;

    while (areaIndices){
        int i = firstBitSet(areaIndices) - 1;

        setAreas.x += splittingSets.x & (KDNode::bitmap(1)<<i) ? areas[i] : 0.0f;
        setAreas.y += splittingSets.y & (KDNode::bitmap(1)<<i) ? areas[i] : 0.0f;
        setAreas.z += splittingSets.z & (KDNode::bitmap(1)<<i) ? areas[i] : 0.0f;
        setAreas.w += splittingSets.w & (KDNode::bitmap(1)<<i) ? areas[i] : 0.0f;

        areaIndices -= KDNode::bitmap(1)<<i;
    }
        
    float lowArea = bitcount(splittingSets.x) * setAreas.x + bitcount(splittingSets.y) * setAreas.y;
    float highArea = bitcount(splittingSets.z) * setAreas.z + bitcount(splittingSets.w) * setAreas.w;

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

template <bool useIndices>
__global__ void 
__launch_bounds__(96) 
    CalcSAH(int *upperLeafIDs,
            char *info,
            float *splitPoss,
            int *primitiveIndex, KDNode::bitmap *primBitmap,
            float *nodeSurface,
            float* primAreas,
            float4 *aabbMin, float4 *aabbMax,
            KDNode::bitmap4 *splitTriangleSet,
            float2 *childAreas,
            KDNode::bitmap2 *childSets,
            int *splitSides){
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
        
    if (id < d_activeNodeRange){
        const int parentID = useIndices ? upperLeafIDs[id] : d_activeNodeIndex + id;

        const int primIndex = primitiveIndex[parentID];
        const KDNode::bitmap primBmp = primBitmap[parentID];
            
        // @OPT. Perhaps the threads can fill the area array coalesced?
        float* area = SharedMemory<float>();
        area += TriangleNode::MAX_LOWER_SIZE * threadIdx.x;
        //float area[32];

        KDNode::bitmap bitmap = primBmp;
        while(bitmap){
            int index = firstBitSet(bitmap) - 1;
            area[index] = primAreas[primIndex + index];
            bitmap -= KDNode::bitmap(1)<<index;
        }            

        // @OPT. Calculate the area for bitmap pairs? Then areas
        // can be summed over 2 bits at a time. Or template to N
        // bits. For fun and profit.

        float optimalArea = fInfinity;
        float leftArea, rightArea;
        KDNode::bitmap leftSet, rightSet;
        char axis;
        int splitIndex;

        KDNode::bitmap triangles = primBmp;
        while(triangles){
            int i = firstBitSet(triangles) - 1;

            CalcAreaForSets(splitTriangleSet[primIndex + i], KDNode::X,
                            primIndex + i,
                            primBmp, area, 
                            optimalArea, 
                            leftArea, rightArea,
                            leftSet, rightSet, axis, splitIndex);

            CalcAreaForSets(splitTriangleSet[d_triangles + primIndex + i], KDNode::Y,
                            primIndex + i,
                            primBmp, area, 
                            optimalArea, 
                            leftArea, rightArea,
                            leftSet, rightSet, axis, splitIndex);

            CalcAreaForSets(splitTriangleSet[2 * d_triangles + primIndex + i], KDNode::Z,
                            primIndex + i,
                            primBmp, area, 
                            optimalArea, 
                            leftArea, rightArea,
                            leftSet, rightSet, axis, splitIndex);

            triangles -= KDNode::bitmap(1)<<i;
        }
            
        float nodeArea = nodeSurface[parentID];
        bool split = optimalArea < (bitcount(primBmp) - traverselCost) * nodeArea;
        if (split){
            // Dump stuff and move on.
            childAreas[id] = make_float2(leftArea, rightArea);
            childSets[id] = KDNode::make_bitmap2(leftSet, rightSet);
            float3 splitPositions;
            if (splitIndex & 1<<31){
                // A high splitplane was used
                splitPositions = make_float3(aabbMax[splitIndex ^ 1<<31]);
            }else{
                // A low splitplane was used
                splitPositions = make_float3(aabbMin[splitIndex]);
            }
            splitPoss[parentID] = axis == KDNode::X ? splitPositions.x : (axis == KDNode::Y ? splitPositions.y : splitPositions.z);
        }
        info[parentID] = split ? axis : KDNode::LEAF;
        splitSides[id] = split;
    }
}

template <bool useIndices>
__global__ void CreateLowerSAHChildren(int *upperLeafIDs,
                                       int *childSplit,
                                       int *childAddrs,
                                       float2 *childAreas,
                                       KDNode::bitmap2 *childSets,
                                       float* nodeArea,
                                       int* primitiveIndex, KDNode::bitmap* primitiveBitmap,
                                       int2 *children, 
                                       int nodeSplits){

    // @OPT 'or' the childSets onto float4 nodeArea. That way we
    // can get everything in one store/lookup?
        
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
        
    if (id < d_activeNodeRange){
        int split = childSplit[id];

        if (split){
            float2 childrenArea = childAreas[id];
            KDNode::bitmap2 childrenSet = childSets[id];
                
            const int childOffset = childAddrs[id];

            const int parentID = useIndices ? upperLeafIDs[id] : d_activeNodeIndex + id;
            int parentPrimIndex = primitiveIndex[parentID];
                
            const int leftChildID = useIndices 
                ? d_activeNodeIndex + childOffset 
                : d_activeNodeIndex + d_activeNodeRange + childOffset;
            nodeArea[leftChildID] = childrenArea.x;
            primitiveIndex[leftChildID] = parentPrimIndex;
            primitiveBitmap[leftChildID] = childrenSet.x;
                
            const int rightChildID = leftChildID + nodeSplits;
            nodeArea[rightChildID] = childrenArea.y;
            primitiveIndex[rightChildID] = parentPrimIndex;
            primitiveBitmap[rightChildID] = childrenSet.y;

            children[parentID] = make_int2(leftChildID, rightChildID);
        }
    }        
}

template <bool useIndices>
__global__ void PropagateAabbToChildren(int *indices,
                                        char *nodeInfo, float *splitPoss,
                                        float4 *aabbMins, float4 *aabbMaxs,
                                        int2 *children){

    const int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < d_activeNodeRange){
        const int parentID = useIndices ? indices[id] : d_activeNodeIndex + id;

        const char axis = nodeInfo[parentID] & 3;
        if (axis != KDNode::LEAF){
            const float splitPos = splitPoss[parentID];
            
            const int2 childIDs = children[parentID];
            
            const float4 aabbMin = aabbMins[parentID];
            
            aabbMins[childIDs.x] = aabbMin;
            aabbMins[childIDs.y] = make_float4(axis == KDNode::X ? splitPos : aabbMin.x,
                                               axis == KDNode::Y ? splitPos : aabbMin.y,
                                               axis == KDNode::Z ? splitPos : aabbMin.z,
                                               0.0f);
            
            const float4 aabbMax = aabbMaxs[parentID];
            
            aabbMaxs[childIDs.x] = make_float4(axis == KDNode::X ? splitPos : aabbMax.x,
                                               axis == KDNode::Y ? splitPos : aabbMax.y,
                                               axis == KDNode::Z ? splitPos : aabbMax.z,
                                               0.0f);
            aabbMaxs[childIDs.y] = aabbMax;
        }    
    }
}

__global__ void TrimChildBitmaps(int* primitiveIndex,
                                 KDNode::bitmap* primitiveBitmap,
                                 float4 *nodeAabbMin, float4 *nodeAabbMax, 
                                 int* primIndices,
                                 float4* v0s, float4* v1s, float4* v2s, 
                                 int children){
    const int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < children){
        const int primIndex = primitiveIndex[id];
        KDNode::bitmap primBmp = primitiveBitmap[id];

        const float3 nodeMin = make_float3(nodeAabbMin[id]);
        const float3 nodeMax = make_float3(nodeAabbMax[id]);

        KDNode::bitmap triangles = primBmp;
        while(triangles){
            int i = firstBitSet(triangles) - 1;

            int indice = primIndices[primIndex + i];
            
            const float3 v0 = make_float3(v0s[indice]);
            const float3 v1 = make_float3(v1s[indice]);
            const float3 v2 = make_float3(v2s[indice]);

            if (!TriangleAabbIntersection(v0, v1, v2, nodeMin, nodeMax))
                primBmp -= KDNode::bitmap(1)<<i;
            
            triangles -= KDNode::bitmap(1)<<i;
        }

        primitiveBitmap[id] = primBmp;

    }
}

#endif
