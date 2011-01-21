// Raytracer interface for CUDA
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _I_RAY_TRACER_H_
#define _I_RAY_TRACER_H_

#include <Math/Math.h>
#include <Resources/CUDA/CUDADataBlock.h>
#include <Utils/CUDA/IntersectionTests.h>
#include <Utils/CUDA/Utils.h>

namespace OpenEngine {
    namespace Display {
        class IRenderCanvas;
    }
    namespace Scene {
        class TriangleNode;
    }
    namespace Utils {
        namespace CUDA {
            
            class IRayTracer {
            public:
                enum IntersectionAlgorithm {MOELLER, WOOP};

            protected:
                bool visualizeRays;
                IntersectionAlgorithm intersectionAlgorithm;
                bool leafSkipping;
                bool printTiming;
                float renderTime;
                
                Resources::CUDA::CUDADataBlock<1, float4> *origin;
                Resources::CUDA::CUDADataBlock<1, float4> *direction;
                int screenWidth;
                
            public:
                IRayTracer();
                virtual ~IRayTracer();

                virtual void Trace(Display::IRenderCanvas* canvas, uchar4* canvasData) = 0;
                virtual void HostTrace(int x, int y, Scene::TriangleNode* nodes) = 0;

                void SetVisualizeRays(const bool v) {visualizeRays = v;}
                bool GetVisualizeRays() const { return visualizeRays; }
                void SetIntersectionAlgorithm(const IntersectionAlgorithm a) { intersectionAlgorithm = a;}
                IntersectionAlgorithm GetIntersectionAlgorithm() const { return intersectionAlgorithm; }
                void SetLeafSkipping(const bool s) {leafSkipping = s;}
                bool GetLeafSkipping() const { return leafSkipping; }
                void PrintTiming(const bool p) { printTiming = p;}
                bool GetPrintTiming() const { return printTiming; }
                float GetRenderTime() const { return renderTime; }

#define PW 4
#define PH 8
                static inline __device__ __host__ int PacketIndex(int id, int screenWidth){
                    int cell = id / (PW * PH);
                    int bx = cell % (screenWidth / PW);
                    int by = cell / (screenWidth / PW);
                    int x = id % PW;
                    int y = (id % (PW * PH)) / PW;
                    return bx * PW + x + (by * PH + y) * screenWidth;
                }

                static inline __host__ __device__
                float WoopLambda(const float3 origin, const float3 direction,
                                 const float4 m2){
                    return - (dot(make_float3(m2), origin) - m2.w) / dot(make_float3(m2), direction);
                }
                
                static inline __host__ __device__
                float WoopUV(const float3 origin, const float3 direction,
                             const float lambda, const float4 m0){
    
                        return lambda * dot(make_float3(m0), direction) + dot(make_float3(m0), origin) - m0.w;
                }

                static inline __device__ __host__ 
                void Woop(float4* woop0, float4* woop1, float4* woop2, int prim,
                          float3 origin, float3 direction, int &primHit, float3 &tHit){
                    
                    float3 hitCoords;
                    const float4 woop = FetchGlobalData(woop2, prim);
                    hitCoords.x = WoopLambda(origin, direction, woop);
                    if (0.0f <= hitCoords.x && hitCoords.x <= tHit.x){
                        const float4 w0 = FetchGlobalData(woop0, prim);
                        hitCoords.y = WoopUV(origin, direction, hitCoords.x, w0);
                        const float4 w1 = FetchGlobalData(woop1, prim);
                        hitCoords.z = WoopUV(origin, direction, hitCoords.x, w1);
                        
                        if (hitCoords.y >= -0.0f && hitCoords.z >= -0.0f && hitCoords.y + hitCoords.z <= 1.0f){
                            primHit = prim;
                            tHit = hitCoords;
                        }
                    }
                }
                
                static inline __device__ __host__ 
                void MoellerTrumbore(float4* v0s, float4* v1s, float4* v2s, int prim,
                                     float3 origin, float3 direction, int &primHit, float3 &tHit){
                    
                    const float3 v0 = make_float3(FetchGlobalData(v0s, prim));
                    const float3 v1 = make_float3(FetchGlobalData(v1s, prim));
                    const float3 v2 = make_float3(FetchGlobalData(v2s, prim));

                    const float3 e1 = v1 - v0;
                    const float3 e2 = v2 - v0;
                    const float3 p = cross(direction, e2);
                    const float3 t = origin - v0;
                    const float3 q = cross(t, e1);

#ifdef __CUDA_ARCH__
                    const float invDet = __fdividef(1.0f, dot(p, e1));
#else
                    const float invDet = 1.0f / dot(p, e1);
#endif

                    float3 hitCoords;
                    hitCoords.x = invDet * dot(q, e2);
                    if (0.0f <= hitCoords.x && hitCoords.x <= tHit.x){
                        hitCoords.y = invDet * dot(p, t);
                        hitCoords.z = invDet * dot(q, direction);
                        
                        bool hit = hitCoords.y >= 0.0f && hitCoords.z >= 0.0f 
                            && hitCoords.y + hitCoords.z <= 1.0f;
                        
                        if (hit){
                            primHit = prim;
                            tHit = hitCoords;
                        }
                    }
                }

                template <bool invDir>
                static inline __device__ __host__ 
                bool RayBoxIntersection(const float3 ori, const float3 dir,
                                        float3 boxMin, float3 boxMax){

                    CUDALogger("RayBoxIntersection(" << ori << ", " << dir << ", " << boxMin << ", " << boxMax << ")");
                    
                    boxMin = invDir ? (boxMin - ori) * dir : (boxMin - ori) / dir;
                    boxMax = invDir ? (boxMax - ori) * dir : (boxMax - ori) / dir;
                    
                    float nearAlpha = min(boxMin.x, boxMax.x);
                    nearAlpha = max(nearAlpha, min(boxMin.y, boxMax.y));
                    nearAlpha = max(nearAlpha, min(boxMin.z, boxMax.z));
                    
                    float farAlpha = max(boxMax.x, boxMin.x);
                    farAlpha = min(farAlpha, max(boxMin.y, boxMax.y));
                    farAlpha = min(farAlpha, max(boxMin.z, boxMax.z));

                    CUDALogger("RayBoxIntersection result: " << (nearAlpha <= farAlpha));

                    return nearAlpha <= farAlpha;
                }
                
            protected:
                void CreateInitialRays(Display::IRenderCanvas* canvas);
                void RenderRays(uchar4 *canvas, int rays);
            };

        }
    }
}

#endif
