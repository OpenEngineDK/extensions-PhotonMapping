// CUDA Primitive Utils.
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS)
//
// This program is free software; It is covered by the GNU General
// Public License version 2 or any later version.
// See the GNU General Public License for more details (see LICENSE).
//--------------------------------------------------------------------

#ifndef _CUDA_PRIMITIVE_UTILS_H_
#define _CUDA_PRIMITIVE_UTILS_H_

template <class T> T __host__ __device__ make(float t0, float t1, float t2, float t3);

template <>
float __host__ __device__ make(float t0, float t1, float t2, float t3){
    return t0;
}
template <>
float2 __host__ __device__ make(float t0, float t1, float t2, float t3){
    return make_float2(t0, t1);
}
template <>
float3 __host__ __device__ make(float t0, float t1, float t2, float t3){
    return make_float3(t0, t1, t2);
}
template <>
float4 __host__ __device__ make(float t0, float t1, float t2, float t3){
    return make_float4(t0, t1, t2, t3);
}

#endif
