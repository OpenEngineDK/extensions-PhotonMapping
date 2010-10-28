// Kernels for creating triangle upper nodes children and splitting the triangles
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Meta/CUDA.h>
#include <Utils/CUDA/Kernels/PhotonMapDeviceVars.h>

namespace OpenEngine {
namespace Utils {
namespace CUDA {
namespace Kernels {

    __host__ __device__ float Volume(float3 min, float3 max){
        return (max.x - min.x) * (max.y - min.y) * (max.z - min.z);
    }

    /*
    __host__ __device__ void Sort6(float* es){

        int e = 0;
        e = es[e] < es[1] ? e : 1;
        e = es[e] < es[2] ? e : 2;
        e = es[e] < es[3] ? e : 3;
        e = es[e] < es[4] ? e : 4;
        e = es[e] < es[5] ? e : 5;
        float t = es[e];
        es[e] = es[0];
        es[0] = t;

        e = 1;
        e = es[e] < es[2] ? e : 2;
        e = es[e] < es[3] ? e : 3;
        e = es[e] < es[4] ? e : 4;
        e = es[e] < es[5] ? e : 5;
        t = es[e];
        es[e] = es[1];
        es[1] = t;
        
        e = 2;
        e = es[e] < es[3] ? e : 3;
        e = es[e] < es[4] ? e : 4;
        e = es[e] < es[5] ? e : 5;
        t = es[e];
        es[e] = es[2];
        es[2] = t;

        e = 3;
        e = es[e] < es[4] ? e : 4;
        e = es[e] < es[5] ? e : 5;
        t = es[e];
        es[e] = es[3];
        es[3] = t;

        // e = 4;
        // e = es[e] < es[5] ? e : 5;
        // t = es[e];
        // es[e] = es[4];
        // es[4] = t;
    }

    __global__ void AdjustBoundingBox(float4* aabbMins, float4* aabbMaxs,
                                      float4* p0s, float4* p1s, float4* p2s,
                                      float4 *origMins, float4 *origMaxs,
                                      float *origArea,
                                      int primitives){
        const int id = blockDim.x * blockIdx.x + threadIdx.x;
        
        if (id < primitives){
            float4 hat = aabbMins[id];
            const int primID = hat.w;

            float3 aabbCorners[2];
            aabbCorners[0] = make_float3(hat);
            aabbCorners[1] = make_float3(aabbMaxs[id]);

            float3 ps[3];
            ps[0] = make_float3(p0s[primID]);
            ps[1] = make_float3(p1s[primID]);
            ps[2] = make_float3(p2s[primID]);
            
            float3 intersections[6];

#pragma unroll 3
            for (int i = 0; i < 3; ++i){
                int j = (i + 1) % 3;

                float alphas[6];
                alphas[0] = (aabbCorners[0].x - ps[i].x) / (ps[j].x - ps[i].x);
                alphas[1] = (aabbCorners[0].y - ps[i].y) / (ps[j].y - ps[i].y);
                alphas[2] = (aabbCorners[0].z - ps[i].z) / (ps[j].z - ps[i].z);
                alphas[3] = (aabbCorners[1].x - ps[i].x) / (ps[j].x - ps[i].x);
                alphas[4] = (aabbCorners[1].y - ps[i].y) / (ps[j].y - ps[i].y);
                alphas[5] = (aabbCorners[1].z - ps[i].z) / (ps[j].z - ps[i].z);

                // Sort 
                Sort6(alphas);

                // find the third and fourth smallest alpha, which
                // constitutes intersections inside the bounding box.
                intersections[i*2] = (1.0f - alphas[2]) * ps[i] - alphas[2] * ps[j];
                intersections[i*2+1] = (1.0f - alphas[3]) * ps[i] - alphas[3] * ps[j];
            }

            float3 tempCorners[] = {intersections[0], intersections[0]};
#pragma unroll 6
            for (int i = 1; i < 6; ++i){
                tempCorners[0] = min(tempCorners[0], intersections[i]);
                tempCorners[1] = max(tempCorners[1], intersections[i]);
            }
            
            aabbCorners[0] = max(aabbCorners[0], tempCorners[0]);
            aabbCorners[1] = min(aabbCorners[1], tempCorners[1]);

            float3 origMin = make_float3(origMins[primID]);
            float3 origMax = make_float3(origMaxs[primID]);
            
            float origVolume = Volume(origMin, origMax);
            float newVolume = Volume(aabbCorners[0], aabbCorners[1]);

            float newArea = sqrt(newVolume / origVolume) * origArea[primID];
            newArea = min(newVolume, newArea);

            aabbMins[id] = make_float4(aabbCorners[0], primID);
            aabbMaxs[id] = make_float4(aabbCorners[1], newArea);
        }

    }
*/
    __device__ bool TightTriangleBB(float3 p0, float3 p1, float3 p2,
                                             float3 &aabbMin, float3 &aabbMax){
        float3 tempMin = aabbMax;
        float3 tempMax = aabbMin;

        // p0 -> p1 intersections
        float3 dir = p1 - p0;
        float3 minInters = (aabbMin - p0) / dir;
        float3 maxInters = (aabbMax - p0) / dir;
        
        float3 minAlphas = min(minInters, maxInters);
        float3 maxAlphas = max(minInters, maxInters);

        float nearAlpha = max(minAlphas.x, max(minAlphas.y, minAlphas.z));
        float3 inter = p0 + nearAlpha * dir;
        if (aabbMin.x <= inter.x && inter.x <= aabbMax.x &&
            aabbMin.y <= inter.y && inter.y <= aabbMax.y &&
            aabbMin.z <= inter.z && inter.z <= aabbMax.z){
            tempMin = min(tempMin, inter);
            tempMax = max(tempMax, inter);
        }

        float farAlpha = min(maxAlphas.x, min(maxAlphas.y, maxAlphas.z));
        inter = p0 + farAlpha * dir;
        if (aabbMin.x <= inter.x && inter.x <= aabbMax.x &&
            aabbMin.y <= inter.y && inter.y <= aabbMax.y &&
            aabbMin.z <= inter.z && inter.z <= aabbMax.z){
            tempMin = min(tempMin, inter);
            tempMax = max(tempMax, inter);
        }

        // p1 -> p2 intersections
        dir = p2 - p1;
        minInters = (aabbMin - p1) / dir;
        maxInters = (aabbMax - p1) / dir;
        
        minAlphas = min(minInters, maxInters);
        maxAlphas = max(minInters, maxInters);        
        
        nearAlpha = max(minAlphas.x, max(minAlphas.y, minAlphas.z));
        farAlpha = min(maxAlphas.x, min(maxAlphas.y, maxAlphas.z));

        nearAlpha = max(minAlphas.x, max(minAlphas.y, minAlphas.z));
        inter = p1 + nearAlpha * dir;
        if (aabbMin.x <= inter.x && inter.x <= aabbMax.x &&
            aabbMin.y <= inter.y && inter.y <= aabbMax.y &&
            aabbMin.z <= inter.z && inter.z <= aabbMax.z){
            tempMin = min(tempMin, inter);
            tempMax = max(tempMax, inter);
        }

        farAlpha = min(maxAlphas.x, min(maxAlphas.y, maxAlphas.z));
        if (0.0f <= farAlpha && farAlpha <= 1.0f){
            float3 inter = p1 + farAlpha * dir;
            if (aabbMin.x <= inter.x && inter.x <= aabbMax.x &&
                aabbMin.y <= inter.y && inter.y <= aabbMax.y &&
                aabbMin.z <= inter.z && inter.z <= aabbMax.z){
                tempMin = min(tempMin, inter);
                tempMax = max(tempMax, inter);
            }
        }

        // p2 -> p0 intersections
        dir = p0 - p2;
        minInters = (aabbMin - p2) / dir;
        maxInters = (aabbMax - p2) / dir;
        
        minAlphas = min(minInters, maxInters);
        maxAlphas = max(minInters, maxInters);        
        
        nearAlpha = max(minAlphas.x, max(minAlphas.y, minAlphas.z));
        farAlpha = min(maxAlphas.x, min(maxAlphas.y, maxAlphas.z));

        nearAlpha = max(minAlphas.x, max(minAlphas.y, minAlphas.z));
        inter = p2 + nearAlpha * dir;
        if (aabbMin.x <= inter.x && inter.x <= aabbMax.x &&
            aabbMin.y <= inter.y && inter.y <= aabbMax.y &&
            aabbMin.z <= inter.z && inter.z <= aabbMax.z){
            tempMin = min(tempMin, inter);
            tempMax = max(tempMax, inter);
        }

        farAlpha = min(maxAlphas.x, min(maxAlphas.y, maxAlphas.z));
        inter = p2 + farAlpha * dir;
        if (aabbMin.x <= inter.x && inter.x <= aabbMax.x &&
            aabbMin.y <= inter.y && inter.y <= aabbMax.y &&
            aabbMin.z <= inter.z && inter.z <= aabbMax.z){
            tempMin = min(tempMin, inter);
                tempMax = max(tempMax, inter);
        }

        aabbMin = max(aabbMin, tempMin);
        aabbMax = min(aabbMax, tempMax);

        return aabbMin.x <= aabbMax.x && aabbMin.y <= aabbMax.y && aabbMin.z <= aabbMax.z;
    }

    __host__ bool TightTriangleBB(float3 p0, float3 p1, float3 p2,
                                  float3 &aabbMin, float3 &aabbMax, bool hat){
        float3 tempMin = aabbMax;
        float3 tempMax = aabbMin;

        bool hit = false;

        // p0 -> p1 intersections
        float3 dir = p1 - p0;
        float3 minInters = (aabbMin - p0) / dir;
        float3 maxInters = (aabbMax - p0) / dir;
        
        float3 minAlphas = min(minInters, maxInters);
        float3 maxAlphas = max(minInters, maxInters);

        float nearAlpha = max(minAlphas.x, max(minAlphas.y, minAlphas.z));
        float3 inter = p0 + nearAlpha * dir;
        if (aabbMin.x <= inter.x && inter.x <= aabbMax.x &&
            aabbMin.y <= inter.y && inter.y <= aabbMax.y &&
            aabbMin.z <= inter.z && inter.z <= aabbMax.z){
            logger.info << "inter0: " << Convert::ToString(p0) << 
                " + " << Utils::Convert::ToString(nearAlpha) << 
                " * " << Convert::ToString(dir) << 
                " = " << Convert::ToString(inter) << logger.end;
            tempMin = min(tempMin, inter);
            tempMax = max(tempMax, inter);
        }

        float farAlpha = min(maxAlphas.x, min(maxAlphas.y, maxAlphas.z));
        inter = p0 + farAlpha * dir;
        if (aabbMin.x <= inter.x && inter.x <= aabbMax.x &&
            aabbMin.y <= inter.y && inter.y <= aabbMax.y &&
            aabbMin.z <= inter.z && inter.z <= aabbMax.z){
            logger.info << "inter1: " << Convert::ToString(p0) << 
                " + " << Utils::Convert::ToString(farAlpha) << 
                " * " << Convert::ToString(dir) << 
                " = " << Convert::ToString(inter) << logger.end;
            tempMin = min(tempMin, inter);
            tempMax = max(tempMax, inter);
            hit = true;
        }

        // p1 -> p2 intersections
        dir = p2 - p1;
        logger.info << "dir: " << Convert::ToString(dir) << logger.end;
        minInters = (aabbMin - p1) / dir;
        maxInters = (aabbMax - p1) / dir;
        
        minAlphas = min(minInters, maxInters);
        maxAlphas = max(minInters, maxInters);        

        logger.info << "minAlphas: " << Convert::ToString(minAlphas) << logger.end;
        logger.info << "maxAlphas: " << Convert::ToString(maxAlphas) << logger.end;
        
        nearAlpha = max(minAlphas.x, max(minAlphas.y, minAlphas.z));
        farAlpha = min(maxAlphas.x, min(maxAlphas.y, maxAlphas.z));

        nearAlpha = max(minAlphas.x, max(minAlphas.y, minAlphas.z));
        inter = p1 + nearAlpha * dir;
            logger.info << "inter2: " << Convert::ToString(p1) << 
                " + " << Utils::Convert::ToString(nearAlpha) << 
                " * " << Convert::ToString(dir) << 
                " = " << Convert::ToString(inter) << logger.end;
        if (aabbMin.x <= inter.x && inter.x <= aabbMax.x &&
            aabbMin.y <= inter.y && inter.y <= aabbMax.y &&
            aabbMin.z <= inter.z && inter.z <= aabbMax.z){
            logger.info << "inter2: " << Convert::ToString(p1) << 
                " + " << Utils::Convert::ToString(nearAlpha) << 
                " * " << Convert::ToString(dir) << 
                " = " << Convert::ToString(inter) << logger.end;
            tempMin = min(tempMin, inter);
            tempMax = max(tempMax, inter);
            hit = true;
        }

        farAlpha = min(maxAlphas.x, min(maxAlphas.y, maxAlphas.z));
        inter = p1 + farAlpha * dir;
            logger.info << "inter3: " << Convert::ToString(p1) << 
                " + " << Utils::Convert::ToString(farAlpha) << 
                " * " << Convert::ToString(dir) << 
                " = " << Convert::ToString(inter) << logger.end;
        if (aabbMin.x <= inter.x && inter.x <= aabbMax.x &&
            aabbMin.y <= inter.y && inter.y <= aabbMax.y &&
            aabbMin.z <= inter.z && inter.z <= aabbMax.z){
            logger.info << "inter3: " << Convert::ToString(p1) << 
                " + " << Utils::Convert::ToString(farAlpha) << 
                " * " << Convert::ToString(dir) << 
                " = " << Convert::ToString(inter) << logger.end;
            tempMin = min(tempMin, inter);
            tempMax = max(tempMax, inter);
            hit = true;
        }

        // p2 -> p0 intersections
        dir = p0 - p2;
        minInters = (aabbMin - p2) / dir;
        maxInters = (aabbMax - p2) / dir;
        
        minAlphas = min(minInters, maxInters);
        maxAlphas = max(minInters, maxInters);        
        
        nearAlpha = max(minAlphas.x, max(minAlphas.y, minAlphas.z));
        farAlpha = min(maxAlphas.x, min(maxAlphas.y, maxAlphas.z));

        nearAlpha = max(minAlphas.x, max(minAlphas.y, minAlphas.z));
        inter = p2 + nearAlpha * dir;
        if (aabbMin.x <= inter.x && inter.x <= aabbMax.x &&
            aabbMin.y <= inter.y && inter.y <= aabbMax.y &&
            aabbMin.z <= inter.z && inter.z <= aabbMax.z){
            logger.info << "inter4: " << Convert::ToString(p2) << 
                " + " << Utils::Convert::ToString(nearAlpha) << 
                " * " << Convert::ToString(dir) << 
                " = " << Convert::ToString(inter) << logger.end;
            tempMin = min(tempMin, inter);
            tempMax = max(tempMax, inter);
            hit = true;
        }

        farAlpha = min(maxAlphas.x, min(maxAlphas.y, maxAlphas.z));
        inter = p2 + farAlpha * dir;
        if (aabbMin.x <= inter.x && inter.x <= aabbMax.x &&
            aabbMin.y <= inter.y && inter.y <= aabbMax.y &&
            aabbMin.z <= inter.z && inter.z <= aabbMax.z){
            logger.info << "inter5: " << Convert::ToString(p2) << 
                " + " << Utils::Convert::ToString(farAlpha) << 
                " * " << Convert::ToString(dir) << 
                " = " << Convert::ToString(inter) << logger.end;
            tempMin = min(tempMin, inter);
            tempMax = max(tempMax, inter);
            hit = true;
        }

        aabbMin = max(aabbMin, tempMin);
        aabbMax = min(aabbMax, tempMax);

        //return aabbMin.x <= aabbMax.x && aabbMin.y <= aabbMax.y && aabbMin.z <= aabbMax.z;
        return hit;
    }

    __global__ void AdjustBoundingBox2(float4* aabbMins, float4* aabbMaxs,
                                       float4* p0s, float4* p1s, float4* p2s,
                                       float4 *origMins, float4 *origMaxs,
                                       float *origArea,
                                       int primitives){

        const int id = blockDim.x * blockIdx.x + threadIdx.x;
        
        if (id < primitives){
            float4 hat = aabbMins[id];
            const int primID = hat.w;
            float3 minCorner = make_float3(hat);
            float3 maxCorner = make_float3(aabbMaxs[id]);

            float3 ps[3];
            ps[0] = make_float3(p0s[primID]);
            ps[1] = make_float3(p1s[primID]);
            ps[2] = make_float3(p2s[primID]);

            bool hit = TightTriangleBB(ps[0], ps[1], ps[2], minCorner, maxCorner);
                
            float3 origMin = make_float3(origMins[primID]);
            float3 origMax = make_float3(origMaxs[primID]);
            
            float origVolume = Volume(origMin, origMax);
            float newVolume = Volume(minCorner, maxCorner);

            float newArea = sqrt(newVolume / origVolume) * origArea[primID];
            newArea = hit ? min(newVolume, newArea) : 0.0f;

            aabbMins[id] = make_float4(minCorner, primID);
            aabbMaxs[id] = make_float4(maxCorner, newArea);
        }
    }

}
}
}
}
