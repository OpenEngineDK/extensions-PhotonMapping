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
#include <Utils/CUDA/IntersectionTests.h>

namespace OpenEngine {
    using namespace Display;
    using namespace Resources;
    using namespace Resources::CUDA;
    namespace Utils {
        namespace CUDA {

#include <Utils/CUDA/Kernels/ColorKernels.h>

            __constant__ int d_rays;

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

#define MAX_PRIMS 512
#define MAX_THREADS 128
#define MIN_BLOCKS 2
            
            template <bool useWoop>
            __global__ void 
            __launch_bounds__(MAX_THREADS, MIN_BLOCKS) 
            BruteTracing(float4* origins, float4* directions,
                         float4 *v0s, float4 *v1s, float4 *v2s,
                         float4 *n0s, float4 *n1s, float4 *n2s,
                         uchar4 *c0s,
                         uchar4 *canvas,
                         int prims){
                const int id = blockDim.x * blockIdx.x + threadIdx.x;
                
                if (id < d_rays){

                    prims = prims < MAX_PRIMS ? prims : MAX_PRIMS;

                    float3 origin = make_float3(origins[id]);
                    float3 dir = make_float3(directions[id]);
                    
                    float3 tHit;

                    float4 color = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    
                    do {
                        tHit.x = fInfinity;

                        int primHit = -1;
                        for (int prim = 0; prim < prims; ++prim){

                            if (useWoop){
                                IRayTracer::Woop(v0s, v1s, v2s, prim,
                                                 origin, dir, primHit, tHit);
                            }else{
                                IRayTracer::MoellerTrumbore(v0s, v1s, v2s, prim,
                                                            origin, dir, primHit, tHit);
                            }
                        }
                        
                        if (tHit.x < fInfinity){
                            float4 newColor = Lighting(tHit, origin, dir, 
                                                       n0s, n1s, n2s,
                                                       c0s, primHit);
                            
                            color = BlendColor(color, newColor);
                        }
                    } while(tHit.x < fInfinity && color.w < 0.97f);
                    
                    canvas[id] = make_uchar4(color.x * 255, color.y * 255, color.z * 255, color.w * 255);
                }
            }

            void BruteTracer::Trace(Display::IRenderCanvas* canvas, uchar4* canvasData){
                CreateInitialRays(canvas);

                int height = canvas->GetHeight();
                int width = canvas->GetWidth();
                
                int rays = height * width;

                cudaMemcpyToSymbol(d_rays, &rays, sizeof(int));

                if (visualizeRays){
                    RenderRays(canvasData, rays);
                    return;
                }

                START_TIMER(timerID);
                if (intersectionAlgorithm == WOOP){
                    float4 *woop0, *woop1, *woop2;
                    geom->GetWoopValues(&woop0, &woop1, &woop2);

                    KernelConf conf = KernelConf1D(rays, MAX_THREADS);
                    BruteTracing<true><<<conf.blocks, conf.threads>>>
                        (origin->GetDeviceData(), direction->GetDeviceData(),
                         woop0, woop1, woop2,
                         geom->GetNormal0Data(), geom->GetNormal1Data(), geom->GetNormal2Data(),
                         geom->GetColor0Data(),
                         canvasData,
                         geom->GetSize());
                    if (printTiming) PRINT_TIMER(timerID, "Brute tracing using Woop");

                }else{
                    KernelConf conf = KernelConf1D(rays, 64);
                    if (printTiming) START_TIMER(timerID);
                    //logger.info << "BruteTracing<<<" << blocks << ", " << threads << ", " << smemSize << ">>>" << logger.end;
                    BruteTracing<false><<<conf.blocks, conf.threads>>>
                        (origin->GetDeviceData(), direction->GetDeviceData(),
                         geom->GetP0Data(), geom->GetP1Data(), geom->GetP2Data(), 
                         geom->GetNormal0Data(), geom->GetNormal1Data(), geom->GetNormal2Data(),
                         geom->GetColor0Data(),
                         canvasData,
                         geom->GetSize());
                    if (printTiming) PRINT_TIMER(timerID, "Brute tracing using Möller-Trumbore");
                }
                cudaThreadSynchronize();
                cutStopTimer(timerID);
                renderTime = cutGetTimerValue(timerID);
                CHECK_FOR_CUDA_ERROR();
            }

            void BruteTracer::HostTrace(int x, int y, TriangleNode* nodes){
                throw Core::Exception("Not implemented");
            }            

        }
    }
}
