// Kernels for calculating empty space splitting in upper nodes
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

__global__ void CalcEmptySpaceSplits(float4 *propagatedAabbMin,
                                     float4 *propagatedAabbMax,
                                     float4 *reducedAabbMin,
                                     float4 *reducedAabbMax,
                                     char *emptySpacePlanes,
                                     int *emptySpaceNodes){

    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < d_activeNodeRange){
        int nodesRequired = 0;
        char planes = 0;

        const float3 pMin = make_float3(propagatedAabbMin[id]);
        const float3 pMax = make_float3(propagatedAabbMax[id]);

        const float3 rMin = make_float3(reducedAabbMin[id]);
        
        bool isEmptySpace = rMin.x - pMin.x > d_emptySpaceThreshold * (pMax.x - pMin.x);
        if (isEmptySpace){
            planes |= 1<<0;
            nodesRequired += 2;
        }

        isEmptySpace = rMin.y - pMin.y > d_emptySpaceThreshold * (pMax.y - pMin.y);
        if (isEmptySpace){
            planes |= 1<<1;
            nodesRequired += 2;
        }

        isEmptySpace = rMin.z - pMin.z > d_emptySpaceThreshold * (pMax.z - pMin.z);
        if (isEmptySpace){
            planes |= 1<<2;
            nodesRequired += 2;
        }

        const float3 rMax = make_float3(reducedAabbMax[id]);

        isEmptySpace = pMax.x - rMax.x > d_emptySpaceThreshold * (pMax.x - pMin.x);
        if (isEmptySpace){
            planes |= 1<<3;
            nodesRequired += 2;
        }
        
        isEmptySpace = pMax.y - rMax.y > d_emptySpaceThreshold * (pMax.y - pMin.y);
        if (isEmptySpace){
            planes |= 1<<4;
            nodesRequired += 2;
        }

        isEmptySpace = pMax.z - rMax.z > d_emptySpaceThreshold * (pMax.z - pMin.z);
        if (isEmptySpace){
            planes |= 1<<5;
            nodesRequired += 2;
        }
        
        emptySpacePlanes[id] = planes;
        emptySpaceNodes[id] = nodesRequired;
        if (planes) d_createdEmptySplits = true;
    }
}

__global__ void CreateEmptyNodes(char* nodeInfo, float* splitPoss,
                                 float4 *aabbMins, float4 *aabbMaxs,
                                 int2* primitiveInfo, int2* children, 
                                 char* emptySpacePlanes, int* emptySpaceAddrs,
                                 int* indices){

    const int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < d_activeNodeRange){
        int parentID = d_activeNodeIndex + id;

        const char planes = emptySpacePlanes[parentID];

        //@OPT provide early out for threads with no planes? They just
        //need to write their indice?

        // Values to be propagated to the final child
        const char info = nodeInfo[parentID];
        const float splitPos = splitPoss[parentID];

        // @OPT more coalescence if the right indices start midway
        // into the array?
        int leftID = parentID + d_activeNodeRange + emptySpaceAddrs[id];
        int rightID = leftID+1;

        const float3 aabbMin = make_float3(aabbMins[parentID]);

        if (planes & 1){ // min.x splits, ie the right child holds primitives
            nodeInfo[parentID] = KDNode::X;
            splitPoss[parentID] = aabbMin.x;
            nodeInfo[leftID] = KDNode::LEAF; // @OPT set empty bit in info?
            primitiveInfo[leftID] = make_int2(0, 0);
            parentID = rightID;
        }

        if (planes & 2){ // min.y splits, ie the right child holds primitives
            nodeInfo[parentID] = KDNode::Y;
            splitPoss[parentID] = aabbMin.y;
            nodeInfo[leftID] = KDNode::LEAF; // @OPT set empty bit in info?
            primitiveInfo[leftID] = make_int2(0, 0);
            parentID = rightID;
        }

        if (planes & 4){ // min.z splits, ie the right child holds primitives
            nodeInfo[parentID] = KDNode::Z;
            splitPoss[parentID] = aabbMin.z;
            nodeInfo[leftID] = KDNode::LEAF; // @OPT set empty bit in info?
            primitiveInfo[leftID] = make_int2(0, 0);
            parentID = rightID;
        }

        const float3 aabbMax = make_float3(aabbMaxs[parentID]);

        if (planes & 8){ // max.x splits, ie the left child holds primitives
            nodeInfo[parentID] = KDNode::X;
            splitPoss[parentID] = aabbMax.x;
            nodeInfo[rightID] = KDNode::LEAF; // @OPT set empty bit in info?
            primitiveInfo[rightID] = make_int2(0, 0);
            parentID = leftID;
        }

        if (planes & 16){ // max.y splits, ie the left child holds primitives
            nodeInfo[parentID] = KDNode::Y;
            splitPoss[parentID] = aabbMax.y;
            nodeInfo[rightID] = KDNode::LEAF;
            primitiveInfo[rightID] = make_int2(0, 0);
            parentID = leftID;
        }

        if (planes & 32){ // max.z splits, ie the left child holds primitives
            nodeInfo[parentID] = KDNode::Z;
            splitPoss[parentID] = aabbMax.z;
            nodeInfo[rightID] = KDNode::LEAF;
            primitiveInfo[rightID] = make_int2(0, 0);
            parentID = leftID;
        }

        indices[id] = parentID;
        nodeInfo[parentID] = info;
        splitPoss[parentID] = splitPos;
        primitiveInfo[parentID] = primitiveInfo[d_activeNodeIndex + id];
        aabbMins[parentID] = make_float4(aabbMin, 0.0f);
        aabbMaxs[parentID] = make_float4(aabbMax, 0.0f);
    }
}

template <bool useIndices>
__global__ void PropagateChildAabb(int *indices,
                                   char *nodeInfo, float *splitPoss,
                                   float4 *aabbMins, float4 *aabbMaxs,
                                   int2 *children){

    const int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < d_activeNodeRange){
        const int parentID = useIndices ? indices[id] : d_activeNodeIndex + id;

        const char axis = nodeInfo[parentID] & 3;
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

