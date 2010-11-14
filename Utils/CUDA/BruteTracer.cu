// Brute force ray tracer for CUDA
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Utils/CUDA/BruteTracer.h>

#include <Display/IRenderCanvas.h>
#include <Display/IViewingVolume.h>
#include <Utils/CUDA/GeometryList.h>
#include <Utils/CUDA/SharedMemory.h>
#include <Utils/CUDA/Utils.h>

namespace OpenEngine {
    using namespace Display;
    using namespace Resources;
    using namespace Resources::CUDA;
    namespace Utils {
        namespace CUDA {

            BruteTracer::BruteTracer(GeometryList* geom)
                : geom(geom) {

                cutCreateTimer(&timerID);
            }

            BruteTracer::~BruteTracer() {}
            
            __constant__ int d_rays;
            __constant__ int d_screenHeight;
            __constant__ int d_screenWidth;

            __global__ void BruteTracing(float4* origins, float4* directions,
                                         float4 *v0s, float4 *v1s, float4 *v2s,
                                         uchar4 *c0,
                                         uchar4 *canvas,
                                         int prims){
                const int id = blockDim.x * blockIdx.x + threadIdx.x;
                
                if (id < d_rays){

                    float3 *v0 = SharedMemory<float3>();
                    float3 *v1 = v0 + blockDim.x;
                    float3 *v2 = v1 + blockDim.x;
                    
                    float3 origin = make_float3(origins[id]);
                    float3 dir = make_float3(directions[id]);
                    
                    float tHit = fInfinity;
                    int primHit = 0;
                    uchar4 color = make_uchar4(0, 0, 0, 0);

                    for (int prim = 0; prim < prims; ++prim){
                        float3 hitCoords;
                        bool hit = TriangleRayIntersection(make_float3(v0s[prim]), make_float3(v1s[prim]), make_float3(v2s[prim]), 
                                                           origin, dir, hitCoords);
                        
                        if (hit && hitCoords.x < tHit){
                            primHit = prim;
                            tHit = hitCoords.x;
                        }
                    }

                    if (tHit < fInfinity)
                        canvas[id] = c0[primHit];
                    else
                        canvas[id] = color;
                }
            }

            void BruteTracer::Trace(Display::IRenderCanvas* canvas, uchar4* canvasData){
                CreateInitialRays(canvas);

                int height = canvas->GetHeight();
                int width = canvas->GetWidth();
                
                int rays = height * width;

                cudaMemcpyToSymbol(d_screenWidth, &width, sizeof(int));
                cudaMemcpyToSymbol(d_screenHeight, &height, sizeof(int));
                cudaMemcpyToSymbol(d_rays, &rays, sizeof(int));

                if (visualizeRays){
                    RenderRays(canvasData, rays);
                    return;
                }

                unsigned int blocks, threads;
                Calc1DKernelDimensions(rays, blocks, threads, 128);
                int smemSize = threads * sizeof(float3) * 3;
                //START_TIMER(timerID);
                //logger.info << "BruteTracing<<<" << blocks << ", " << threads << ", " << smemSize << ">>>" << logger.end;
                BruteTracing<<<blocks, threads, smemSize>>>(origin->GetDeviceData(), direction->GetDeviceData(),
                                                            geom->GetP0Data(), geom->GetP1Data(), geom->GetP2Data(), 
                                                            geom->GetColor0Data(),
                                                            canvasData,
                                                            geom->GetSize());
                //PRINT_TIMER(timerID, "Brute tracing");
                CHECK_FOR_CUDA_ERROR();
                
            }

        }
    }
}
