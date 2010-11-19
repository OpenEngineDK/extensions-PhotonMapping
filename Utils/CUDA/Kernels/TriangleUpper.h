// Kernels for segmenting triangle upper nodes
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

__global__ void AddIndexToAabb(float4 *aabbIn,
                               int size,
                               float4 *aabbOut){
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
        
    if (id < size){
        float4 aabb = aabbIn[id];
        aabb.w = id;
        aabbOut[id] = aabb;
    }

}

__global__ void ExtractIndexFromAabb(float4 *aabbIn,
                                     int *out, int size){
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
        
    if (id < size){
        float4 aabb = aabbIn[id];
        out[id] = aabb.w;
    }

}
