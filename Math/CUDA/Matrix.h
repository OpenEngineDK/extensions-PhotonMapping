// Math matrix
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS)
//
// This program is free software; It is covered by the GNU General
// Public License version 2 or any later version.
// See the GNU General Public License for more details (see LICENSE).
//--------------------------------------------------------------------

#ifndef _CUDA_MATRIX_H_
#define _CUDA_MATRIX_H_

#include <Meta/CUDA.h>
#include <Math/Matrix.h>
#include <Utils/CUDA/PrimitiveUtils.h>

namespace OpenEngine {
namespace Math {
namespace CUDA {

    class Matrix44f {
    private:
        float4 elm[4];

    public:
        __host__ __device__ Matrix44f() {
            elm[0] = make_float4(1.0f,0.0f,0.0f,0.0f);
            elm[1] = make_float4(0.0f,1.0f,0.0f,0.0f);
            elm[2] = make_float4(0.0f,0.0f,1.0f,0.0f);
            elm[3] = make_float4(0.0f,0.0f,0.0f,1.0f);
        }

        __host__ Matrix44f(const Matrix<4,4, float> m) {
            for (unsigned int i = 0; i < 4; ++i){
                Vector<4, float> row = m.GetRow(i);
                elm[i] = make_float4(row.Get(0), row.Get(1), row.Get(2), row.Get(3));
            }
        }

        const float4 __host__ __device__ GetRow(const int i) const {
            return elm[i];
        }
        
        const float4 __host__ __device__ operator*(const float4 v) const{
            return make_float4(dot(elm[0], v),
                               dot(elm[1], v),
                               dot(elm[2], v),
                               dot(elm[3], v));
        }
    };

    class Matrix33f {
    private:
        float3 elm[3];

    public:
        __host__ __device__ Matrix33f() {
            elm[0] = make_float3(1.0f,0.0f,0.0f);
            elm[1] = make_float3(0.0f,1.0f,0.0f);
            elm[2] = make_float3(0.0f,0.0f,1.0f);
        }

        __host__ __device__ Matrix33f(const Matrix44f m){
            for (int i = 0; i < 3; ++i)
                elm[i] = make_float3(m.GetRow(i));
        }

        __host__ Matrix33f(const Matrix<3,3, float> m) {
            for (unsigned int i = 0; i < 3; ++i){
                Vector<3, float> row = m.GetRow(i);
                elm[i] = make_float3(row.Get(0), row.Get(1), row.Get(2));
            }
        }

        const float3 __host__ __device__ operator*(const float3 v) const{
            return make_float3(dot(elm[0], v),
                               dot(elm[1], v),
                               dot(elm[2], v));
        }
    };
}
}
}

#endif
