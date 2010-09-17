// The point typedef and functions that goes with it.
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _CUDA_POINT_H_
#define _CUDA_POINT_H_

#include <Meta/CUDA.h>

typedef float4 point;


inline __host__ __device__ point make_point(float val){
    //return make_float3(val);
    return make_float4(val, val, val, 1.0f);
}

inline __host__ __device__ point make_point(float x, float y, float z){
    //return make_float3(x, y, z);
    return make_float4(x, y, z, 1.0f);
}

inline __host__ __device__ point pointMin(point p1, point p2){
    return make_point(min(p1.x, p2.x),
                      min(p1.y, p2.y),
                      min(p1.z, p2.z));
}

inline __host__ __device__ point pointMax(point p1, point p2){
    return make_point(max(p1.x, p2.x),
                      max(p1.y, p2.y),
                      max(p1.z, p2.z));
}

inline __host__ __device__ bool aabbContains(point aabbMin, point aabbMax, point p){
    return aabbMin.x <= p.x && p.x <= aabbMax.x &&
        aabbMin.y <= p.y && p.y <= aabbMax.y &&
        aabbMin.z <= p.z && p.z <= aabbMax.z;
}

#endif
