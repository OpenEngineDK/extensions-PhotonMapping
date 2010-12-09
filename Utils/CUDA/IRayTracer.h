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

#include <Resources/CUDA/CUDADataBlock.h>
#include <Utils/CUDA/IntersectionTests.h>

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
                
                Resources::CUDA::CUDADataBlock<1, float4> *origin;
                Resources::CUDA::CUDADataBlock<1, float4> *direction;
                int screenWidth;
                
            public:
                IRayTracer();
                virtual ~IRayTracer();

                virtual void Trace(Display::IRenderCanvas* canvas, uchar4* canvasData) = 0;
                virtual void HostTrace(int x, int y, Scene::TriangleNode* nodes) = 0;

                void SetVisualizeRays(const bool v) {visualizeRays = v;}
                void SetIntersectionAlgorithm(const IntersectionAlgorithm a) { intersectionAlgorithm = a;}

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

                static inline __device__ __host__ 
                void Woop(float4* woop0, float4* woop1, float4* woop2, int prim,
                          float3 origin, float3 direction, int &primHit, float3 &tHit){
                    
                    float3 hitCoords;
                    hitCoords.x = WoopLambda(origin, direction, woop2[prim]);
                    if (0.0f <= hitCoords.x && hitCoords.x < tHit.x){
                        hitCoords.y = WoopUV(origin, direction, hitCoords.x, woop0[prim]);
                        hitCoords.z = WoopUV(origin, direction, hitCoords.x, woop1[prim]);
                        
                        if (hitCoords.y >= 0.0f && hitCoords.z >= 0.0f && hitCoords.y + hitCoords.z <= 1.0f){
                            primHit = prim;
                            tHit = hitCoords;
                        }
                    }
                }
                
                static inline __device__ __host__ 
                void MoellerTrumbore(float4* v0s, float4* v1s, float4* v2s, int prim,
                                     float3 origin, float3 direction, int &primHit, float3 &tHit){
                    
                    float3 hitCoords;

                    const float3 v0 = make_float3(v0s[prim]);
                    const float3 v1 = make_float3(v1s[prim]);
                    const float3 v2 = make_float3(v2s[prim]);

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
                    
                    hitCoords.x = invDet * dot(q, e2);
                    if (0.0f <= hitCoords.x && hitCoords.x < tHit.x){
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
                
            protected:
                void CreateInitialRays(Display::IRenderCanvas* canvas);
                void RenderRays(uchar4 *canvas, int rays);
            };

        }
    }
}

#endif
