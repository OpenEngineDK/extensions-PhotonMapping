// Intersection tests
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef __TRACING_INTERSECTION_TESTS_H_
#define __TRACING_INTERSECTION_TESTS_H_

#include <Utils/CUDA/Utils.h>
#include <Utils/CUDA/Convert.h>
#include <Scene/KDNode.h>

using namespace OpenEngine::Scene;
using namespace OpenEngine::Utils;

#define GAMMA 0.00001f
#define EPSILON 1.0001f

inline __host__ __device__
void a0XTests(const float v0X, const float v0Y, const float v0Z, 
              const float v1X, const float v1Y, const float v1Z,
              const float v2X, const float v2Y, const float v2Z, 
              const float halfSizeX, const float halfSizeY, const float halfSizeZ, 
              bool &ret){

    // a00
    float p0 = v0Z * v1Y - v0Y * v1Z;
    float p1 = (v1Y - v0Y) * v2Z - (v1Z - v0Z) * v2Y;
    float r = halfSizeZ * fabsf(v1Y - v0Y) + halfSizeY * fabsf(v1Z - v0Z) + GAMMA;
    r *= EPSILON;
    ret &= !((p0 > r && p1 > r) || (p0 < -r && p1 < -r));

    CUDALogger("a00: p0: " << p0 << ", p1: " << p1 << ", r: " << r << ", ret: " << ret);

    // a01
    p0 = v1Z * v2Y - v1Y * v2Z;
    p1 = (v2Y - v1Y) * v0Z - (v2Z - v1Z) * v0Y;
    r = halfSizeZ * fabsf(v2Y - v1Y) + halfSizeY * fabsf(v2Z - v1Z) + GAMMA;
    r *= EPSILON;
    ret &= !((p0 > r && p1 > r) || (p0 < -r && p1 < -r));

    CUDALogger("a01: p0: " << p0 << ", p1: " << p1 << ", r: " << r << ", ret: " << ret);

    // a02
    p0 = v2Z * v0Y - v2Y * v0Z;
    p1 = (v0Y - v2Y) * v1Z - (v0Z - v2Z) * v1Y;
    r = halfSizeZ * fabsf(v0Y - v2Y) + halfSizeY * fabsf(v0Z - v2Z) + GAMMA;
    r *= EPSILON;
    ret &= !((p0 > r && p1 > r) || (p0 < -r && p1 < -r));

    CUDALogger("a02: p0: " << p0 << ", p1: " << p1 << ", r: " << r << ", ret: " << ret);
}

inline __host__ __device__
void a0XTests(const float3 v0, const float3 v1, const float3 v2, 
              const float3 halfSize,
              bool &ret){
    a0XTests(v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, v2.x, v2.y, v2.z, 
             halfSize.x, halfSize.y, halfSize.z, 
             ret);
}

inline __host__ __device__
void a1XTests(const float v0X, const float v0Y, const float v0Z, 
              const float v1X, const float v1Y, const float v1Z,
              const float v2X, const float v2Y, const float v2Z, 
              const float halfSizeX, const float halfSizeY, const float halfSizeZ, 
              bool &ret){
    
    // a10
    float p0 = v0X * v1Z - v0Z * v1X;
    float p1 = (v1Z - v0Z) * v2X - (v1X - v0X) * v2Z;
    float r = halfSizeX * fabsf(v1Z - v0Z) + halfSizeZ * fabsf(v1X - v0X) + GAMMA;
    r *= EPSILON;
    ret &= !((p0 > r && p1 > r) || (p0 < -r && p1 < -r));

    CUDALogger("a10: p0: " << p0 << ", p1: " << p1 << ", r: " << r << ", ret: " << ret);
    
    // a11
    p0 = v1X * v2Z - v1Z * v2X;
    p1 = (v2Z - v1Z) * v0X - (v2X - v1X) * v0Z;
    r = halfSizeX * fabsf(v2Z - v1Z) + halfSizeZ * fabsf(v2X - v1X) + GAMMA;
    r *= EPSILON;
    ret &= !((p0 > r && p1 > r) || (p0 < -r && p1 < -r));

    CUDALogger("a11: p0: " << p0 << ", p1: " << p1 << ", r: " << r << ", ret: " << ret);
    
    // a12
    p0 = v2X * v0Z - v2Z * v0X;
    p1 = (v0Z - v2Z) * v1X - (v0X - v2X) * v1Z;
    r = halfSizeX * fabsf(v0Z - v2Z) + halfSizeZ * fabsf(v0X - v2X) + GAMMA;
    r *= EPSILON;
    ret &= !((p0 > r && p1 > r) || (p0 < -r && p1 < -r));

    CUDALogger("a12: p0: " << p0 << ", p1: " << p1 << ", r: " << r << ", ret: " << ret);
}
inline __host__ __device__
void a1XTests(const float3 v0, const float3 v1, const float3 v2, 
              const float3 halfSize,
              bool &ret){
    a1XTests(v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, v2.x, v2.y, v2.z, 
             halfSize.x, halfSize.y, halfSize.z, 
             ret);
}

inline __host__ __device__
void a2XTests(const float v0X, const float v0Y, const float v0Z, 
              const float v1X, const float v1Y, const float v1Z,
              const float v2X, const float v2Y, const float v2Z, 
              const float halfSizeX, const float halfSizeY, const float halfSizeZ, 
              bool &ret){
    
    // a20
    float p0 = v0Y * v1X - v0X * v1Y;
    float p1 = (v1X - v0X) * v2Y - (v1Y - v0Y) * v2X;
    float r =  halfSizeY * fabsf(v1X - v0X) + halfSizeX * fabsf(v1Y - v0Y) + GAMMA;
    r *= EPSILON;
    ret &= !((p0 > r && p1 > r) || (p0 < -r && p1 < -r));

    CUDALogger("a20: p0: " << p0 << ", p1: " << p1 << ", r: " << r << ", ret: " << ret);
    
    // a21
    p0 = v1Y * v2X - v1X * v2Y;
    p1 = (v2X - v1X) * v0Y - (v2Y - v1Y) * v0X;
    r = halfSizeY * fabsf(v2X - v1X) + halfSizeX * fabsf(v2Y - v1Y) + GAMMA;
    r *= EPSILON;
    ret &= !((p0 > r && p1 > r) || (p0 < -r && p1 < -r));

    CUDALogger("a21: p0: " << p0 << ", p1: " << p1 << ", r: " << r << ", ret: " << ret);
                    
    // a22
    p0 = v2Y * v0X - v2X * v0Y;
    p1 = (v0X - v2X) * v1Y - (v0Y - v2Y) * v1X;
    r = halfSizeY * fabsf(v0X - v2X) + halfSizeX * fabsf(v0Y - v2Y) + GAMMA;
    r *= EPSILON;
    ret &= !((p0 > r && p1 > r) || (p0 < -r && p1 < -r));

    CUDALogger("a22: p0: " << p0 << ", p1: " << p1 << ", r: " << r << ", ret: " << ret);
}
inline __host__ __device__
void a2XTests(const float3 v0, const float3 v1, const float3 v2, 
              const float3 halfSize,
              bool &ret){
    a2XTests(v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, v2.x, v2.y, v2.z, 
             halfSize.x, halfSize.y, halfSize.z, 
             ret);
}


/**
 * Divides the box [aabbMin, aabbMax] along axis at splitPos and
 * checks which sides the triangle [a,b,c] intersects.
 *
 * Asserts that the triangle [a,b,c] intersects the box [aabbMin,
 * aabbMax]. If this is not the case then doom on you!
 */
inline __host__ __device__
void DivideTriangle(const float3 a, const float3 b, const float3 c,
                    const float3 aabbMin, const float3 aabbMax,
                    const char axis, const float splitPos,
                    bool &intersectsLeft, bool &intersectsRight){
    
#ifndef __CUDA_ARCH__
    logger.info << "AABB: " << Utils::CUDA::Convert::ToString(aabbMin) << " -> " << Utils::CUDA::Convert::ToString(aabbMax) << logger.end;
    logger.info << "Triangle: " << Utils::CUDA::Convert::ToString(a) << ", " << Utils::CUDA::Convert::ToString(b) << ", " << Utils::CUDA::Convert::ToString(c) << logger.end;
#endif

    if (axis == KDNode::X){
        const float triMin = min(a.x, min(b.x, c.x));
        const float triMax = max(a.x, max(b.x, c.x));
        
        //intersectsLeft = triMin <= splitPos && aabbMin.x <= triMax;
        //intersectsRight = triMin <= aabbMax.x && splitPos <= triMax;
        intersectsLeft = triMin < splitPos;
        intersectsRight = splitPos < triMax;

        if (intersectsLeft == true && intersectsRight == true){
            // Perform further testing based on step 3 from
            // Akenine-Möller
                
            const float halfSizeY = (aabbMax.y - aabbMin.y) * 0.5f;
            const float halfSizeZ = (aabbMax.z - aabbMin.z) * 0.5f;
            const float centerY = aabbMin.y + halfSizeY;
            const float centerZ = aabbMin.z + halfSizeZ;

            const float v0Y = a.y - centerY;
            const float v0Z = a.z - centerZ;

            const float v1Y = b.y - centerY;
            const float v1Z = b.z - centerZ;

            const float v2Y = c.y - centerY;
            const float v2Z = c.z - centerZ;
                
            if (intersectsLeft){
                const float halfSizeX = (splitPos - aabbMin.x) * 0.5f;
                const float centerX = aabbMin.x + halfSizeX;
                const float v0X = a.x - centerX;
                const float v1X = b.x - centerX;
                const float v2X = c.x - centerX;
                    
                // Skip a00, a01, a02 as they don't contain any reference to X

                a1XTests(v0X, v0Y, v0Z, v1X, v1Y, v1Z, v2X, v2Y, v2Z, 
                         halfSizeX, halfSizeY, halfSizeZ, 
                         intersectsLeft);
                    
                a2XTests(v0X, v0Y, v0Z, v1X, v1Y, v1Z, v2X, v2Y, v2Z, 
                         halfSizeX, halfSizeY, halfSizeZ, 
                         intersectsLeft);
            }
                
            if (intersectsRight){
                const float halfSizeX = (aabbMax.x - splitPos) * 0.5f;
                const float centerX = splitPos + halfSizeX;
                const float v0X = a.x - centerX;
                const float v1X = b.x - centerX;
                const float v2X = c.x - centerX;

                // Skip a00, a01, a02 as they don't contain any reference to X

                a1XTests(v0X, v0Y, v0Z, v1X, v1Y, v1Z, v2X, v2Y, v2Z, 
                         halfSizeX, halfSizeY, halfSizeZ, 
                         intersectsRight);
                    
                a2XTests(v0X, v0Y, v0Z, v1X, v1Y, v1Z, v2X, v2Y, v2Z, 
                         halfSizeX, halfSizeY, halfSizeZ, 
                         intersectsRight);
            }
        }
    }else if (axis == KDNode::Y){
        const float triMin = min(a.y, min(b.y, c.y));
        const float triMax = max(a.y, max(b.y, c.y));
        
        //intersectsLeft = triMin <= splitPos && aabbMin.y <= triMax;
        //intersectsRight = triMin <= aabbMax.y && splitPos <= triMax;
        intersectsLeft = triMin < splitPos;
        intersectsRight = splitPos < triMax;

        if (intersectsLeft && intersectsRight){
            // Perform further testing based on step 3 from
            // Akenine-Möller

            const float halfSizeX = (aabbMax.x - aabbMin.x) * 0.5f;
            const float halfSizeZ = (aabbMax.z - aabbMin.z) * 0.5f;
            const float centerX = aabbMin.x + halfSizeX;
            const float centerZ = aabbMin.z + halfSizeZ;

            const float v0X = a.x - centerX;
            const float v0Z = a.z - centerZ;

            const float v1X = b.x - centerX;
            const float v1Z = b.z - centerZ;

            const float v2X = c.x - centerX;
            const float v2Z = c.z - centerZ;

            if (intersectsLeft){
                const float halfSizeY = (splitPos - aabbMin.y) * 0.5f;
                const float centerY = aabbMin.y + halfSizeY;
                const float v0Y = a.y - centerY;
                const float v1Y = b.y - centerY;
                const float v2Y = c.y - centerY;
                    
                a0XTests(v0X, v0Y, v0Z, v1X, v1Y, v1Z, v2X, v2Y, v2Z, 
                         halfSizeX, halfSizeY, halfSizeZ, 
                         intersectsLeft);
                    
                // Skip a10, a11, a12 as they don't contain any reference to Y

                a2XTests(v0X, v0Y, v0Z, v1X, v1Y, v1Z, v2X, v2Y, v2Z, 
                         halfSizeX, halfSizeY, halfSizeZ, 
                         intersectsLeft);                    
            }

            if (intersectsRight){
                const float halfSizeY = (aabbMax.y - splitPos) * 0.5f;
                const float centerY = splitPos + halfSizeY;
                const float v0Y = a.y - centerY;
                const float v1Y = b.y - centerY;
                const float v2Y = c.y - centerY;
                    
                a0XTests(v0X, v0Y, v0Z, v1X, v1Y, v1Z, v2X, v2Y, v2Z, 
                         halfSizeX, halfSizeY, halfSizeZ, 
                         intersectsRight);
                    
                // Skip a10, a11, a12 as they don't contain any reference to Y

                a2XTests(v0X, v0Y, v0Z, v1X, v1Y, v1Z, v2X, v2Y, v2Z, 
                         halfSizeX, halfSizeY, halfSizeZ, 
                         intersectsRight);                    
            }

        }
    }else if (axis == KDNode::Z) {
        const float triMin = min(a.z, min(b.z, c.z));
        const float triMax = max(a.z, max(b.z, c.z));
            
        //intersectsLeft = triMin <= splitPos && aabbMin.z <= triMax;
        //intersectsRight = triMin <= aabbMax.z && splitPos <= triMax;
        intersectsLeft = triMin < splitPos;
        intersectsRight = splitPos < triMax;

        if (intersectsLeft && intersectsRight){
            // Perform further testing based on step 3 from
            // Akenine-Möller

            const float halfSizeX = (aabbMax.x - aabbMin.x) * 0.5f;
            const float halfSizeY = (aabbMax.y - aabbMin.y) * 0.5f;
            const float centerX = aabbMin.x + halfSizeX;
            const float centerY = aabbMin.y + halfSizeY;

            const float v0X = a.x - centerX;
            const float v0Y = a.y - centerY;

            const float v1X = b.x - centerX;
            const float v1Y = b.y - centerY;

            const float v2X = c.x - centerX;
            const float v2Y = c.y - centerY;
                
            if (intersectsLeft){
                const float halfSizeZ = (splitPos - aabbMin.z) * 0.5f;
                const float centerZ = aabbMin.z + halfSizeZ;
                const float v0Z = a.z - centerZ;
                const float v1Z = b.z - centerZ;
                const float v2Z = c.z - centerZ;

                a0XTests(v0X, v0Y, v0Z, v1X, v1Y, v1Z, v2X, v2Y, v2Z, 
                         halfSizeX, halfSizeY, halfSizeZ, 
                         intersectsLeft);
                    
                a1XTests(v0X, v0Y, v0Z, v1X, v1Y, v1Z, v2X, v2Y, v2Z, 
                         halfSizeX, halfSizeY, halfSizeZ, 
                         intersectsLeft);

                // Skip a20, a21, a22 as they don't contain any reference to Z
            }

            if (intersectsRight){
                const float halfSizeZ = (aabbMax.z - splitPos) * 0.5f;
                const float centerZ = splitPos + halfSizeZ;
                const float v0Z = a.z - centerZ;
                const float v1Z = b.z - centerZ;
                const float v2Z = c.z - centerZ;

                a0XTests(v0X, v0Y, v0Z, v1X, v1Y, v1Z, v2X, v2Y, v2Z, 
                         halfSizeX, halfSizeY, halfSizeZ, 
                         intersectsRight);
                    
                a1XTests(v0X, v0Y, v0Z, v1X, v1Y, v1Z, v2X, v2Y, v2Z, 
                         halfSizeX, halfSizeY, halfSizeZ, 
                         intersectsRight);

                // Skip a20, a21, a22 as they don't contain any reference to Z
            }
        }
    }
}

inline __host__ __device__ 
bool TriangleAabbIntersectionStep1(const float3 v0, const float3 v1, const float3 v2, 
                                   const float3 aabbMin, const float3 aabbMax){
    
    const float3 triMin = min(v0, min(v1, v2));
    const float3 triMax = max(v0, max(v1, v2));

    return triMin.x <= aabbMax.x && aabbMin.x <= triMax.x
        && triMin.y <= aabbMax.y && aabbMin.y <= triMax.y
        && triMin.z <= aabbMax.z && aabbMin.z <= triMax.z;
}

inline __host__ __device__ 
bool TriangleAabbIntersectionStep3(float3 v0, float3 v1, float3 v2, 
                                   const float3 aabbMin, const float3 aabbMax){

    const float3 halfSize = (aabbMax - aabbMin) * 0.5f;
    const float3 center = aabbMin + halfSize;

    v0 -= center; v1 -= center; v2 -= center;

    // Only test 3 from Akenine-Möller

    bool res = true;
    a0XTests(v0, v1, v2, halfSize, res);

    a1XTests(v0, v1, v2, halfSize, res);

    a2XTests(v0, v1, v2, halfSize, res);

    return res;
}

inline __host__ __device__ bool TriangleAabbIntersection(const float3 v0, const float3 v1, const float3 v2, 
                                                         const float3 aabbMin, const float3 aabbMax){

    return TriangleAabbIntersectionStep1(v0, v1, v2, aabbMin, aabbMax)
        && TriangleAabbIntersectionStep3(v0, v1, v2, aabbMin, aabbMax);
}

/* Möller Trumbore Ray Triangle Intersection */

inline __host__ __device__ bool TriangleRayIntersection(const float3 v0, const float3 v1, const float3 v2,
                                                        const float3 origin, const float3 direction,
                                                        float3 &hit){
    
    const float3 e1 = v1 - v0;
    const float3 e2 = v2 - v0;
    const float3 p = cross(direction, e2);
    const float det = dot(p, e1);
    const float3 t = origin - v0;
    const float3 q = cross(t, e1);

    // if det is 'equal' to zero, the ray lies in the triangles plane
    // and cannot be seen.
    //if (det == 0.0f) return false;

#ifdef __CUDA_ARCH__
    const float invDet = __fdividef(1.0f, det);
#else
    const float invDet = 1.0f / det;
#endif

    hit.x = invDet * dot(q, e2);
    hit.y = invDet * dot(p, t);
    hit.z = invDet * dot(q, direction);

    return hit.x >= 0.0f && hit.y >= 0.0f && hit.z >= 0.0f && hit.y + hit.z <= 1.0f;
}

/*** Woop Ray Triangle Intersection ***/

inline __host__ __device__
void WoopTransformationMatrix(const float3 a, const float3 b, const float3 c, 
                              float4 &m0, float4 &m1, float4 &m2){
    // Compute transformation matrix
    const float3 invM0 = a-c;
    const float3 invM1 = b-c;
    const float3 invM2 = cross(invM0, invM1)-c;
    
    // Invert the damn matrix man
    const float det = invM0.x * (invM1.y * invM2.z - invM1.z * invM2.y) +
        invM1.x * (invM2.y * invM0.z - invM2.z * invM0.y) +
        invM2.x * (invM0.y * invM1.z - invM0.z * invM1.y);

    const float invDet = 1.0f / det;

    const float3 tmp0 = make_float3(invM1.y * invM2.z - invM1.z * invM2.y,
                                    invM2.x * invM1.z - invM2.z * invM1.x,
                                    invM1.x * invM2.y - invM1.y * invM2.x) * invDet;
    
    const float3 tmp1 = make_float3(invM2.y * invM0.z - invM2.z * invM0.y,
                                    invM0.x * invM2.z - invM0.z * invM2.x,
                                    invM2.x * invM0.y - invM2.y * invM0.x) * invDet;
    
    const float3 tmp2 = make_float3(invM0.y * invM1.z - invM0.z * invM1.y,
                                    invM1.x * invM0.z - invM1.z * invM0.x,
                                    invM0.x * invM1.y - invM0.y * invM1.x) * invDet;

    m0 = make_float4(tmp0, dot(tmp0, c));
    m1 = make_float4(tmp1, dot(tmp1, c));
    m2 = make_float4(tmp2, dot(tmp2, c));
}

inline __host__ __device__
float WoopLambda(const float3 origin, const float3 direction,
                const float4 m2){
    
    return - (dot(make_float3(m2), origin) - m2.w) / dot(make_float3(m2), direction);
}

inline __host__ __device__
float WoopUV(const float3 origin, const float3 direction,
             const float lambda, const float4 m0){
    
    return lambda * dot(make_float3(m0), direction) + dot(make_float3(m0), origin) - m0.w;
}

inline __host__ __device__
bool WoopIntersectionTest(const float3 origin, const float3 direction,
                          const float4 m0, const float4 m1, const float4 m2,
                          float3 &tHit){

#ifndef __CUDA_ARCH__
    logger.info << "origin " << Utils::CUDA::Convert::ToString(origin) << logger.end;
    logger.info << "direction " << Utils::CUDA::Convert::ToString(direction) << logger.end;
    logger.info << "m0 " << Utils::CUDA::Convert::ToString(m0) << logger.end;
    logger.info << "m1 " << Utils::CUDA::Convert::ToString(m1) << logger.end;
    logger.info << "m2 " << Utils::CUDA::Convert::ToString(m2) << logger.end;
#endif

    tHit.x = WoopLambda(origin, direction, m2);
    tHit.y = WoopUV(origin, direction, tHit.x, m0);
    tHit.z = WoopUV(origin, direction, tHit.x, m1);

#ifndef __CUDA_ARCH__
    logger.info << "tHit: " << Utils::CUDA::Convert::ToString(tHit) << logger.end;
#endif
    
    return tHit.x >= 0.0f && tHit.y >= 0.0f && tHit.z >= 0.0f && tHit.y + tHit.z <= 1.0f;
}

inline __host__ __device__
bool WoopIntersection(const float3 a, const float3 b, const float3 c, 
                      const float3 origin, const float3 direction){

    float4 m0, m1, m2;
    WoopTransformationMatrix(a, b, c, m0, m1, m2);
        
    float3 tHit = make_float3(1000.0f, 1000.0f, 1000.0f);
    bool hit = WoopIntersectionTest(origin, direction, m0, m1, m2, tHit);

    return hit;
}


#endif


