// Kernels for segmenting triangle upper nodes
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

__global__ void CalcPrimitiveAabb(float4* v0s, float4* v1s, float4* v2s, 
                                  float4 *primAabbMin, float4 *primAabbMax,
                                  int size){

    const int id = blockDim.x * blockIdx.x + threadIdx.x;
        
    if (id < size){
        float3 minCorner, maxCorner;
        minCorner = maxCorner = make_float3(v0s[id]);
        
        float3 vertex = make_float3(v1s[id]);
        minCorner = min(minCorner, vertex);
        maxCorner = max(maxCorner, vertex);
        
        vertex = make_float3(v2s[id]);
        primAabbMin[id] = make_float4(min(minCorner, vertex), __int_as_float(id));
        primAabbMax[id] = make_float4(max(maxCorner, vertex), 0.0f);
    }
}

__global__ void AddIndexToAabb(float4 *aabbIn,
                               int size,
                               float4 *aabbOut){
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
        
    if (id < size){
        float4 aabb = aabbIn[id];
        aabb.w = __int_as_float(id);
        aabbOut[id] = aabb;
    }

}

__global__ void ExtractIndexFromAabb(float4 *aabbIn,
                                     int *out, int size){
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
        
    if (id < size){
        float4 aabb = aabbIn[id];
        out[id] = __float_as_int(aabb.w);
    }

}
