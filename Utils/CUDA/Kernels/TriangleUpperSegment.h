// Kernels for segmenting triangle upper nodes
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

__global__ void NodeSegments(int2* primitiveInfo,
                             int* nodeSegments){
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
        
    if (id < d_activeNodeRange){
        nodeSegments[id] = 1 + (primitiveInfo[id].y-1) / Segments::SEGMENT_SIZE;
    }
}

__global__ void MarkOwnerStart(int* owners,
                               int* startAddrs){
        
    // Need to add 1 to the id, since the first segments are owned
    // by node 0 and thus shouldn't add to the prefix sum.
    const int id = blockDim.x * blockIdx.x + threadIdx.x + 1;

    if (id < d_activeNodeRange){
        int index = startAddrs[id];
        owners[index] = 1;
    }
    if (id == 1)
        owners[0] = d_activeNodeIndex;
}

__global__ void CalcSegmentPrimitives(int *owners,
                                      int *nodeSegmentAddrs,
                                      int2 *nodePrimInfo,
                                      int2 *segmentPrimInfo){
        
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
        
    if (id < d_segments){
        // @OPT node prim info can be placed in shared memory?
        const int nodeID = owners[id];
        const int offset = Segments::SEGMENT_SIZE * (id - nodeSegmentAddrs[nodeID - d_activeNodeIndex]);
        const int2 primInfo = nodePrimInfo[nodeID];
        const int index = primInfo.x + offset;
        const int range = min(Segments::SEGMENT_SIZE, primInfo.y - offset);
        segmentPrimInfo[id] = make_int2(index, range);
    }
}
