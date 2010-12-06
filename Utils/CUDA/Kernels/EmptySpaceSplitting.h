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

__global__ void EmptySpaceSplitting(char* nodeInfo, float* splitPoss,
                                    KDNode::amount* primitiveAmount,
                                    int* parents, int2* children, 
                                    char* emptySpacePlanes, int* emptySpaceAddrs,
                                    float4 *minPlanes, float4 *maxPlanes,
                                    int emptyNodes){
    const int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < d_activeNodeRange){
        const int childID = d_activeNodeIndex + id;

        // Create empty space planes
        char planes = emptySpacePlanes[id];

        int parentID = parents[id];
        const int2 childIDs = children[parentID];
        
        const int emptyStartAddr = planes ? d_activeNodeIndex - emptyNodes + emptySpaceAddrs[id] : childID;

        // Set the parent to point to the emptySpace node or new child
        // address
        if (childIDs.x == childID - emptyNodes)
            children[parentID].x = emptyStartAddr;
        else 
            children[parentID].y = emptyStartAddr;

        // Early out
        if (planes == 0) return;

        parentID = emptyStartAddr;

        const float3 minPlane = make_float3(minPlanes[id]);

        if (planes & 1){ // min.x splits, ie the right child holds primitives
            planes -= 1;
            nodeInfo[parentID] = KDNode::X;
            splitPoss[parentID] = minPlane.x;
            int leftID = parentID+1;
            int rightID = planes ? leftID+1 : childID;
            children[parentID] = make_int2(leftID, rightID);
            nodeInfo[leftID] = KDNode::LEAF;
            primitiveAmount[leftID] = 0;

            parentID = rightID;
        }

        if (planes & 2){ // min.y splits, ie the right child holds primitives
            planes -= 2;
            nodeInfo[parentID] = KDNode::Y;
            splitPoss[parentID] = minPlane.y;
            int leftID = parentID+1;
            int rightID = planes ? leftID+1 : childID;
            children[parentID] = make_int2(leftID, rightID);
            nodeInfo[leftID] = KDNode::LEAF;
            primitiveAmount[leftID] = 0;

            parentID = rightID;
        }

        if (planes & 4){ // min.z splits, ie the right child holds primitives
            planes -= 4;
            nodeInfo[parentID] = KDNode::Z;
            splitPoss[parentID] = minPlane.z;
            int leftID = parentID+1;
            int rightID = planes ? leftID+1 : childID;
            children[parentID] = make_int2(leftID, rightID);
            nodeInfo[leftID] = KDNode::LEAF;
            primitiveAmount[leftID] = 0;

            parentID = rightID;
        }

        const float3 maxPlane = make_float3(maxPlanes[id]);

        if (planes & 8){ // max.x splits, ie the right child holds primitives
            planes -= 8;
            nodeInfo[parentID] = KDNode::X;
            splitPoss[parentID] = maxPlane.x;
            int rightID = parentID+1;
            int leftID = planes ? rightID+1 : childID;
            children[parentID] = make_int2(leftID, rightID);
            nodeInfo[rightID] = KDNode::LEAF;
            primitiveAmount[rightID] = 0;

            parentID = leftID;
        }

        if (planes & 16){ // max.y splits, ie the right child holds primitives
            planes -= 16;
            nodeInfo[parentID] = KDNode::Y;
            splitPoss[parentID] = maxPlane.y;
            int rightID = parentID+1;
            int leftID = planes ? rightID+1 : childID;
            children[parentID] = make_int2(leftID, rightID);
            nodeInfo[rightID] = KDNode::LEAF;
            primitiveAmount[rightID] = 0;

            parentID = leftID;
        }

        if (planes & 32){ // max.y splits, ie the right child holds primitives
            planes -= 32;
            nodeInfo[parentID] = KDNode::X;
            splitPoss[parentID] = maxPlane.x;
            int rightID = parentID+1;
            int leftID = childID;
            children[parentID] = make_int2(leftID, rightID);
            nodeInfo[rightID] = KDNode::LEAF;
            primitiveAmount[rightID] = 0;
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

