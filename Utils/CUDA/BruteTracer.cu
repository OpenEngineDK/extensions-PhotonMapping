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

            __constant__ int d_rays;
            __constant__ int d_screenHeight;
            __constant__ int d_screenWidth;

            __constant__ float3 d_lightPosition;
            __constant__ float4 d_lightAmbient;
            __constant__ float4 d_lightDiffuse;
            __constant__ float4 d_lightSpecular;

            BruteTracer::BruteTracer(GeometryList* geom)
                : geom(geom) {

                cutCreateTimer(&timerID);

                float3 lightPosition = make_float3(0.0f, 4.0f, 0.0f);
                cudaMemcpyToSymbol(d_lightPosition, &lightPosition, sizeof(float3));
                float4 lightColor = make_float4(1.0f, 0.92f, 0.8f, 1.0f);
                float4 ambient = lightColor * 0.3f;
                cudaMemcpyToSymbol(d_lightAmbient, &ambient, sizeof(float4));
                float4 diffuse = lightColor * 0.7f;
                cudaMemcpyToSymbol(d_lightDiffuse, &diffuse, sizeof(float4));
                float4 specular = lightColor * 0.3f;
                cudaMemcpyToSymbol(d_lightSpecular, &specular, sizeof(float4));
                CHECK_FOR_CUDA_ERROR();
            }

            BruteTracer::~BruteTracer() {}

            __device__ float4 PhongLighting(float4 color, float3 normal, float3 point, float3 origin){

                float3 lightDir = normalize(d_lightPosition - point);
                
                // Diffuse
                float ndotl = dot(lightDir, normal);
                float diffuse = ndotl < 0.0f ? 0.0f : ndotl;

                // Calculate specular
                float3 reflect = 2.0f * dot(normal, lightDir) * normal - lightDir;
                reflect = normalize(reflect);
                float stemp = dot(normalize(origin - point), reflect);
                stemp = stemp < 0.0f ? 0.0f : stemp;
                float specProp = 1.0f - color.w;
                float specular = specProp * pow(stemp, 128.0f * specProp);

                float4 light = (d_lightAmbient +
                                (d_lightDiffuse * diffuse) +
                                (d_lightSpecular * specular));
                
                return clamp(color * light, 0.0f, 1.0f);
            }

            __device__ float4 Lighting(int prim, float3 hitCoords, 
                                       float3 &origin, float3 &direction,
                                       float4 *n0s, float4 *n1s, float4 *n2s,
                                       uchar4 *c0s){

                float3 point = origin + hitCoords.x * direction;

                float3 n0 = make_float3(n0s[prim]);
                float3 n1 = make_float3(n1s[prim]);
                float3 n2 = make_float3(n2s[prim]);
                
                float3 normal = (1 - hitCoords.y - hitCoords.z) * n0 + hitCoords.y * n1 + hitCoords.z * n2;
                normal = normalize(normal);
                
                uchar4 c = c0s[prim];
                float4 color = make_float4(c.x / 255.0f, c.y / 255.0f, c.z / 255.0f, c.w / 255.0f);

                return PhongLighting(color, normal, point, origin);
            }
            
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
                    tHit.x = fInfinity;
                    int primHit = 0;

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
                        /*
                        float3 n0 = make_float3(n0s[primHit]);
                        float3 n1 = make_float3(n1s[primHit]);
                        float3 n2 = make_float3(n2s[primHit]);

                        float3 normal = (1 - tHit.y - tHit.z) * n0 + tHit.y * n1 + tHit.z * n2;
                        normal = normalize(normal);

                        uchar4 c = c0[primHit];
                        float4 color = make_float4(c.x / 255.0f, c.y / 255.0f, c.z / 255.0f, c.w / 255.0f);
                        
                        float3 point = origin + tHit.x * dir;
                        
                        color = PhongLighting(color, normal, point, origin);
                        */
                        float4 color = Lighting(primHit, tHit, origin, dir, 
                                                n0s, n1s, n2s,
                                                c0s);
                        
                        canvas[id] = make_uchar4(color.x * 255, color.y * 255, color.z * 255, color.w * 255);
                    }else
                        canvas[id] = make_uchar4(0, 0, 0, 0);
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
