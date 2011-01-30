// Kernels for creating triangle upper nodes children and splitting the triangles
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------
    
__global__ void 
__launch_bounds__(Segments::SEGMENT_SIZE) 
    SetSplitSide(int2 *segmentPrimInfo,
                 int* segmentOwner,
                 char* info,
                 float *splitPoss,
                 float4* aabbMins,
                 float4* aabbMaxs,
                 int* splitSides){
        
    const int segmentID = blockIdx.x;

    if (segmentID < d_segments){
        const int2 primInfo = segmentPrimInfo[segmentID];
        if (threadIdx.x < primInfo.y){ 
            const int owner = segmentOwner[segmentID];
            const char axis = info[owner];
            const float splitPos = splitPoss[owner];
                
            const int id = primInfo.x + threadIdx.x;

            switch(axis){
            case KDNode::X:
                splitSides[id] = aabbMins[id].x <= splitPos;
                splitSides[id + d_triangles] = splitPos < aabbMaxs[id].x;
                break;
            case KDNode::Y:
                splitSides[id] = aabbMins[id].y <= splitPos;
                splitSides[id + d_triangles] = splitPos < aabbMaxs[id].y;
                break;
            case KDNode::Z:
                splitSides[id] = aabbMins[id].z <= splitPos;
                splitSides[id + d_triangles] = splitPos < aabbMaxs[id].z;
                break;
            }

            /*
            const float4 aabbMin = aabbMins[id];
            float leftPos = axis == KDNode::X ? aabbMin.x : aabbMin.y;
            leftPos = axis == KDNode::Z ? aabbMin.z : leftPos;
            splitSides[id] = leftPos < splitPos;
                
            const float4 aabbMax = aabbMaxs[id];
            float rightPos = axis == KDNode::X ? aabbMax.x : aabbMax.y;
            rightPos = axis == KDNode::Z ? aabbMax.z : rightPos;
            splitSides[id + d_triangles] = splitPos < rightPos;
            */

            if (segmentID == 0 && threadIdx.x == 0)
                splitSides[d_triangles * 2] = 0;
        }
    }
}

template <bool useFast>
__global__ void 
__launch_bounds__(Segments::SEGMENT_SIZE) 
    SetDivideSide(int2 *segmentPrimInfo,
                  int* segmentOwner,
                  char* info, float *splitPoss,
                  float4* primMins, float4* primMaxs,
                  float4* nodeAabbMins, float4* nodeAabbMaxs,
                  float4* v0s, float4* v1s, float4* v2s,
                  int* splitSides){
    
    const int segmentID = blockIdx.x;

    if (segmentID < d_segments){
        const int2 primInfo = segmentPrimInfo[segmentID];
        if (threadIdx.x < primInfo.y){ 
            const int nodeID = segmentOwner[segmentID];
            const char axis = info[nodeID];
            const float splitPos = splitPoss[nodeID];
                
            const int id = primInfo.x + threadIdx.x;

            bool splitLeft, splitRight;
            if (!useFast){
                switch(axis){
                case KDNode::X:
                    splitLeft = primMins[id].x <= splitPos;
                    splitRight = splitPos < primMaxs[id].x;
                    break;
                case KDNode::Y:
                    splitLeft = primMins[id].y <= splitPos;
                    splitRight = splitPos < primMaxs[id].y;
                    break;
                case KDNode::Z:
                    splitLeft = primMins[id].z <= splitPos;
                    splitRight = splitPos < primMaxs[id].z;
                    break;
                }
                
                if (splitLeft && splitRight){
                    const float3 nodeMin = make_float3(nodeAabbMins[nodeID]);
                    const float3 nodeMax = make_float3(nodeAabbMaxs[nodeID]);
                    const int primID = primMins[id].w;
                    const float3 v0 = make_float3(v0s[primID]);
                    const float3 v1 = make_float3(v1s[primID]);
                    const float3 v2 = make_float3(v2s[primID]);
                    
                    splitLeft = TriangleAabbIntersectionStep3(v0, v1, v2, nodeMin,
                                                          make_float3(axis == KDNode::X ? splitPos : nodeMax.x,
                                                                      axis == KDNode::Y ? splitPos : nodeMax.y,
                                                                      axis == KDNode::Z ? splitPos : nodeMax.z));
                    
                    splitRight = TriangleAabbIntersectionStep3(v0, v1, v2,
                                                               make_float3(axis == KDNode::X ? splitPos : nodeMin.x,
                                                                           axis == KDNode::Y ? splitPos : nodeMin.y,
                                                                           axis == KDNode::Z ? splitPos : nodeMin.z),
                                                               nodeMax);
                }
                
                // @TODO Temporary fix! If the triangles are
                // wrongfully rejected on each side then enable both.
                if (splitLeft == splitRight){
                    splitLeft = splitRight = 1;
                }
                
            }else{

                const int primID = primMins[id].w;
                const float3 v0 = make_float3(v0s[primID]);
                const float3 v1 = make_float3(v1s[primID]);
                const float3 v2 = make_float3(v2s[primID]);            
                const float3 nodeMin = make_float3(nodeAabbMins[nodeID]);
                const float3 nodeMax = make_float3(nodeAabbMaxs[nodeID]);
                
                DivideTriangle(v0, v1, v2, nodeMin, nodeMax, axis, splitPos,
                               splitLeft, splitRight);
            }

            splitSides[id] = splitLeft;
            splitSides[id + d_triangles] = splitRight;

            if (segmentID == 0 && threadIdx.x == 0)
                splitSides[d_triangles * 2] = 0;
        }
    }
}

__global__ void CalcNodeChildSize(int* primitiveIndex, KDNode::amount* primitiveAmount,
                                  int *splitAddrs,
                                  int2* childSize){

    const int id = blockDim.x * blockIdx.x + threadIdx.x;
        
    if (id < d_activeNodeRange){
        const int primIndex = primitiveIndex[id];
        const int primAmount = primitiveAmount[id];

        const int leftSize = (splitAddrs[primIndex + primAmount] - splitAddrs[primIndex]);
        const int rightSize = (splitAddrs[primIndex + primAmount + d_triangles] - splitAddrs[primIndex + d_triangles]);

        childSize[id] = make_int2(leftSize, rightSize);

        if (leftSize < TriangleNode::MAX_LOWER_SIZE ||
            rightSize < TriangleNode::MAX_LOWER_SIZE) d_createdLeafs = true;
    }
}

__global__ void 
__launch_bounds__(Segments::SEGMENT_SIZE) 
    SetPrimitiveLeafSide(int2* segmentPrimInfo,
                         int *segmentOwner,
                         int2 *childSize,
                         int *splitSide,
                         int *leafSide){
        
    const int segmentID = blockIdx.x;

    if (segmentID < d_segments){
        int2 primInfo = segmentPrimInfo[segmentID];
        if (threadIdx.x < primInfo.y){
            const int owner = segmentOwner[segmentID];
            const int2 size = childSize[owner - d_activeNodeIndex];

            const int id = primInfo.x + threadIdx.x;
                
            leafSide[id] = (size.x < TriangleNode::MAX_LOWER_SIZE) * splitSide[id];
            leafSide[id + d_triangles] = (size.y < TriangleNode::MAX_LOWER_SIZE) * splitSide[id + d_triangles];
        }
    }
}

// @OPT include in CalcNodeChildSize? Wait until we construct the
// entire tree to test this.
__global__ void MarkNodeLeafs(int2 *childSize,
                              int *leafNodes){

    const int id = blockDim.x * blockIdx.x + threadIdx.x;
        
    if (id < d_activeNodeRange){
        const int2 size = childSize[id];

        leafNodes[id] = size.x < TriangleNode::MAX_LOWER_SIZE;
        leafNodes[id + d_activeNodeRange] = size.y < TriangleNode::MAX_LOWER_SIZE;
    }
}

template <bool useIndices>
__global__ void CreateUpperChildren(int *indices,
                                    int *primitiveIndex,
                                    KDNode::amount *primitiveAmount,
                                    int2 *childSize,
                                    int *splitAddrs,
                                    int2 *children,
                                    int *parent,
                                    int childStartAddr){

    const int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < d_activeNodeRange){
        const int parentID = useIndices ? indices[id] : d_activeNodeIndex + id;

        const int leftID = childStartAddr + id;
        const int rightID = leftID + d_activeNodeRange;

        const int primIndex = primitiveIndex[parentID];
            
        primitiveIndex[leftID] = splitAddrs[primIndex];
        primitiveIndex[rightID] = splitAddrs[primIndex + d_triangles];

        const int2 size = childSize[id];
        primitiveAmount[leftID] = size.x;
        primitiveAmount[rightID] = size.y;

        children[parentID] = make_int2(leftID, rightID);

        parent[leftID] = parent[rightID] = parentID;
    }
}

template <bool useIndices>
__global__ void CreateUpperChildren(int* indices,
                                    int *primitiveIndex,
                                    KDNode::amount *primitiveAmount,
                                    int2 *childSize,
                                    int *splitAddrs,
                                    int *leafAddrs,
                                    int *leafNodeAddrs,
                                    int2 *children,
                                    int *parent,
                                    int upperLeafPrimitives,
                                    int childStartAddr){

    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < d_activeNodeRange){
            
        const int2 size = childSize[id];

        const int parentID = useIndices ? indices[id] : d_activeNodeIndex + id;

        const int primIndex = primitiveIndex[parentID];
            
        bool isLeaf = size.x < TriangleNode::MAX_LOWER_SIZE;
        int leafAddr = leafNodeAddrs[id];
        int nonLeafAddr = id - leafAddr + d_leafNodes;
        const int leftID = (isLeaf ? leafAddr : nonLeafAddr) + childStartAddr;

        primitiveIndex[leftID] = isLeaf ? leafAddrs[primIndex] + upperLeafPrimitives : splitAddrs[primIndex] - leafAddrs[primIndex];
        primitiveAmount[leftID] = size.x;
        parent[leftID] = parentID;

        id += d_activeNodeRange;

        isLeaf = size.y < TriangleNode::MAX_LOWER_SIZE;
        leafAddr = leafNodeAddrs[id];
        nonLeafAddr = id - leafAddr + d_leafNodes;
        const int rightID = (isLeaf ? leafAddr : nonLeafAddr) + childStartAddr;

        primitiveIndex[rightID] = isLeaf ? leafAddrs[primIndex + d_triangles] + upperLeafPrimitives : splitAddrs[primIndex + d_triangles] - leafAddrs[primIndex + d_triangles];
        primitiveAmount[rightID] = size.y;
        parent[rightID] = parentID;

        children[parentID] = make_int2(leftID, rightID);
    }
}

__global__ void 
__launch_bounds__(Segments::SEGMENT_SIZE) 
    SplitTriangles(int2 *segmentPrimInfo,
                   int *segmentOwners,
                   char *info, float *splitPositions,
                   int *splitSides, int *splitAddrs, 
                   float4* oldAabbMin, float4* oldAabbMax,
                   float4* newAabbMin, float4* newAabbMax){

    const int segmentID = blockIdx.x;
        
    if (segmentID < d_segments){
        int2 primInfo = segmentPrimInfo[segmentID];
        if (threadIdx.x < primInfo.y){
            const int owner = segmentOwners[segmentID];
            const char axis = info[owner];
            const float splitPos = splitPositions[owner];

            int triangleID = threadIdx.x + primInfo.x;
            const float4 aabbMin = oldAabbMin[triangleID];
            const float4 aabbMax = oldAabbMax[triangleID];

            int left = splitSides[triangleID];
            if (left == 1){
                int addr = splitAddrs[triangleID];
                newAabbMin[addr] = aabbMin;
                newAabbMax[addr] = make_float4(axis == KDNode::X ? min(splitPos, aabbMax.x) : aabbMax.x,
                                               axis == KDNode::Y ? min(splitPos, aabbMax.y) : aabbMax.y,
                                               axis == KDNode::Z ? min(splitPos, aabbMax.z) : aabbMax.z,
                                               aabbMax.w);
            }

            int right = splitSides[triangleID + d_triangles];
            if (right == 1){
                int addr = splitAddrs[triangleID + d_triangles];
                newAabbMin[addr] = make_float4(axis == KDNode::X ? max(splitPos, aabbMin.x) : aabbMin.x,
                                               axis == KDNode::Y ? max(splitPos, aabbMin.y) : aabbMin.y,
                                               axis == KDNode::Z ? max(splitPos, aabbMin.z) : aabbMin.z,
                                               aabbMin.w);
                newAabbMax[addr] = aabbMax;
            }

        }
    }
}

__global__ void 
__launch_bounds__(Segments::SEGMENT_SIZE) 
    SplitTriangles(int2 *segmentPrimInfo,
                   int *segmentOwners,
                   char *info, float *splitPositions,
                   int *splitSides, int *splitAddrs, 
                   int *leafSides, int *leafAddrs, 
                   float4* oldAabbMin, float4* oldAabbMax,
                   float4* newAabbMin, float4* newAabbMax,
                   float4* finalAabbMin, float4* finalAabbMax){

    const int segmentID = blockIdx.x;
        
    if (segmentID < d_segments){
        int2 primInfo = segmentPrimInfo[segmentID];
        if (threadIdx.x < primInfo.y){

            const int owner = segmentOwners[segmentID];
            const char axis = info[owner];
            const float splitPos = splitPositions[owner];

            int triangleID = threadIdx.x + primInfo.x;
            const float4 aabbMin = oldAabbMin[triangleID];
            const float4 aabbMax = oldAabbMax[triangleID];

            int left = splitSides[triangleID];
            if (left == 1){
                int leaf = leafSides[triangleID];
                float4* resAabbMin = leaf ? finalAabbMin : newAabbMin;
                float4* resAabbMax = leaf ? finalAabbMax : newAabbMax;
                int addr = leaf ? leafAddrs[triangleID] : splitAddrs[triangleID] - leafAddrs[triangleID];
                resAabbMin[addr] = aabbMin;
                resAabbMax[addr] = make_float4(axis == KDNode::X ? min(splitPos, aabbMax.x) : aabbMax.x,
                                               axis == KDNode::Y ? min(splitPos, aabbMax.y) : aabbMax.y,
                                               axis == KDNode::Z ? min(splitPos, aabbMax.z) : aabbMax.z,
                                               aabbMax.w);
            }

            triangleID += d_triangles;
            int right = splitSides[triangleID];
            if (right == 1){
                int leaf = leafSides[triangleID];
                float4* resAabbMin = leaf ? finalAabbMin : newAabbMin;
                float4* resAabbMax = leaf ? finalAabbMax : newAabbMax;
                int addr = leaf ? leafAddrs[triangleID] : splitAddrs[triangleID] - leafAddrs[triangleID];
                resAabbMin[addr] = make_float4(axis == KDNode::X ? max(splitPos, aabbMin.x) : aabbMin.x,
                                               axis == KDNode::Y ? max(splitPos, aabbMin.y) : aabbMin.y,
                                               axis == KDNode::Z ? max(splitPos, aabbMin.z) : aabbMin.z,
                                               aabbMin.w);
                resAabbMax[addr] = aabbMax;
            }
                
        }
    }
}

__global__ void MarkLeafNodes(int* leafList, 
                              char *nodeInfo,
                              int leafIndex, int newLeafNodes){
        
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
        
    if (id < newLeafNodes){
        leafList[id] = leafIndex + id;
        //nodeInfo[id] = KDNode::LEAF;
    }
}
