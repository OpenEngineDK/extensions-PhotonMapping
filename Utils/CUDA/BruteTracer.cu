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

#include <Utils/CUDA/Kernels/ColorKernels.h>

            __constant__ int d_rays;
            __constant__ int d_screenHeight;
            __constant__ int d_screenWidth;

            BruteTracer::BruteTracer(GeometryList* geom)
                : geom(geom) {

                cutCreateTimer(&timerID);

                float3 lightPosition = make_float3(0.0f, 4.0f, 0.0f);
                cudaMemcpyToSymbol(d_lightPosition, &lightPosition, sizeof(float3));
                float3 lightColor = make_float3(1.0f, 0.92f, 0.8f);
                float3 ambient = lightColor * 0.3f;
                cudaMemcpyToSymbol(d_lightAmbient, &ambient, sizeof(float3));
                float3 diffuse = lightColor * 0.7f;
                cudaMemcpyToSymbol(d_lightDiffuse, &diffuse, sizeof(float3));
                float3 specular = lightColor * 0.3f;
                cudaMemcpyToSymbol(d_lightSpecular, &specular, sizeof(float3));
                CHECK_FOR_CUDA_ERROR();
            }

            BruteTracer::~BruteTracer() {}

            __global__ void BruteTracing(float4* origins, float4* directions,
                                         float4 *v0s, float4 *v1s, float4 *v2s,
                                         float4 *n0s, float4 *n1s, float4 *n2s,
                                         uchar4 *c0s,
                                         uchar4 *canvas,
                                         int prims){
                const int id = blockDim.x * blockIdx.x + threadIdx.x;
                
                if (id < d_rays){

                    /*
                    float3 *v0 = SharedMemory<float3>();
                    float3 *v1 = v0 + blockDim.x;
                    float3 *v2 = v1 + blockDim.x;
                    */
                    
                    float3 origin = make_float3(origins[id]);
                    float3 dir = make_float3(directions[id]);
                    
                    float3 tHit;
                    int primHit = 0;

                    float4 color = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

                    /*
                    for (int i = 0; i < prims; i += blockDim.x){
                        int index = i + threadIdx.x;
                        v0[threadIdx.x] = index < prims ? make_float3(v0s[index]) : make_float3(0.0f);
                        v1[threadIdx.x] = index < prims ? make_float3(v1s[index]) : make_float3(0.0f);
                        v2[threadIdx.x] = index < prims ? make_float3(v2s[index]) : make_float3(0.0f);
                        __syncthreads();
                        
                        for (int prim = 0; prim < blockDim.x; ++prim){
                            float3 hitCoords;
                            bool hit = TriangleRayIntersection(v0[prim], v1[prim], v2[prim], 
                                                               origin, dir, hitCoords);
                            
                            if (hit && hitCoords.x < tHit){
                                primHit = prim;
                                tHit = hitCoords.x;
                            }
                        }
                    }
                    */

                    do {
                        tHit.x = fInfinity;
                        for (int prim = 0; prim < prims; ++prim){
                            float3 hitCoords;
                            bool hit = TriangleRayIntersection(make_float3(v0s[prim]), make_float3(v1s[prim]), make_float3(v2s[prim]), 
                                                               origin, dir, hitCoords);
                            
                            if (hit && hitCoords.x < tHit.x){
                                primHit = prim;
                                tHit = hitCoords;
                            }
                        }
                        
                        if (tHit.x < fInfinity){
                            float4 newColor = Lighting(primHit, tHit, origin, dir, 
                                                n0s, n1s, n2s,
                                                c0s);
                            
                            color = BlendColor(color, newColor);
                        }
                    }while(tHit.x < fInfinity && color.w < 0.97f);
                    
                    canvas[id] = make_uchar4(color.x * 255, color.y * 255, color.z * 255, color.w * 255);
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
                Calc1DKernelDimensions(rays, blocks, threads, 64);
                int smemSize = threads * sizeof(float3) * 3;
                //START_TIMER(timerID);
                //logger.info << "BruteTracing<<<" << blocks << ", " << threads << ", " << smemSize << ">>>" << logger.end;
                BruteTracing<<<blocks, threads, smemSize>>>(origin->GetDeviceData(), direction->GetDeviceData(),
                                                            geom->GetP0Data(), geom->GetP1Data(), geom->GetP2Data(), 
                                                            geom->GetNormal0Data(), geom->GetNormal1Data(), geom->GetNormal2Data(),
                                                            geom->GetColor0Data(),
                                                            canvasData,
                                                            geom->GetSize());
                //PRINT_TIMER(timerID, "Brute tracing");
                CHECK_FOR_CUDA_ERROR();
                
            }

        }
    }
}
