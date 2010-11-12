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

                origin = new CUDADataBlock<1, float4>(1);
                dir = new CUDADataBlock<1, float4>(1);
            }

            BruteTracer::~BruteTracer() {}
            
            __constant__ float3 d_camPos;
            __constant__ int d_rays;
            __constant__ int d_screenHeight;
            __constant__ int d_screenWidth;
            __constant__ float d_ViewProjectionMatrixInverse[16];
            
            __device__ __host__ float3 Unproject(float2 screenPos, float* vpMatInv, float3 origin){
                float4 currentPos = make_float4(vpMatInv[0] * screenPos.x + vpMatInv[4] * screenPos.y + vpMatInv[8] + vpMatInv[12],
                                                vpMatInv[1] * screenPos.x + vpMatInv[5] * screenPos.y + vpMatInv[9] + vpMatInv[13],
                                                vpMatInv[2] * screenPos.x + vpMatInv[6] * screenPos.y + vpMatInv[10] + vpMatInv[14],
                                                vpMatInv[3] * screenPos.x + vpMatInv[7] * screenPos.y + vpMatInv[11] + vpMatInv[15]);
                
                float3 rayEnd = make_float3(currentPos.x / currentPos.w, currentPos.y / currentPos.w, currentPos.z / currentPos.w);
                return normalize(origin - rayEnd);
            }
            
            __global__ void CreateRays(float4* origin,
                                       float4* dir){
                const int id = blockDim.x * blockIdx.x + threadIdx.x;
                
                if (id < d_rays){
                    origin[id] = make_float4(d_camPos, 0.0f);

                    int x = id % d_screenWidth;
                    int y = id / d_screenWidth;
                    float2 screenPos = make_float2((x / (float)d_screenWidth) * 2.0f - 1.0f,
                                                   (y / (float)d_screenHeight) * 2.0f - 1.0f);
                    
                    float3 rayDir = Unproject(screenPos, d_ViewProjectionMatrixInverse, d_camPos);
                    dir[id] = make_float4(rayDir, 0.0f);
                }
            }

            __global__ void RenderRayDir(float4* dir, uchar4 *canvas){
                const int id = blockDim.x * blockIdx.x + threadIdx.x;
                
                if (id < d_rays){
                    float4 d = dir[id] * 0.5f + make_float4(0.5f);
                    canvas[id] = make_uchar4(d.x * 255, d.y * 255, d.z * 255, d.w * 255);
                }
            }

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

                    /*
                    for (int primOffset = 0; primOffset < prims; primOffset += blockDim.x){

                        int primIndex = primOffset + threadIdx.x;
                        v0[threadIdx.x] = primIndex < prims ? make_float3(v0s[primIndex]) : make_float3(1.0f, 0.0f, 0.0f);
                        v1[threadIdx.x] = primIndex < prims ? make_float3(v1s[primIndex]) : make_float3(1.0f, 0.0f, 0.0f);
                        v2[threadIdx.x] = primIndex < prims ? make_float3(v2s[primIndex]) : make_float3(1.0f, 0.0f, 0.0f);
                        __syncthreads();

                        for (int prim = primOffset; prim < primOffset + blockDim.x; ++prim){
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

                    if (tHit < fInfinity)
                        canvas[id] = c0[primHit];
                    else
                        canvas[id] = color;
                }
            }

            void BruteTracer::Trace(Display::IRenderCanvas* canvas, uchar4* canvasData){
                Matrix<4, 4, float> viewProjection = 
                    canvas->GetViewingVolume()->GetViewMatrix() *
                    canvas->GetViewingVolume()->GetProjectionMatrix();
                
                Matrix<4, 4, float> viewProjectionInv = viewProjection.GetInverse();
                Vector<3, float> camPos = canvas->GetViewingVolume()->GetPosition();

                int height = canvas->GetHeight();
                int width = canvas->GetWidth();
                
                int rays = height * width;
                cudaMemcpyToSymbol(d_screenWidth, &width, sizeof(int));
                cudaMemcpyToSymbol(d_screenHeight, &height, sizeof(int));
                cudaMemcpyToSymbol(d_rays, &rays, sizeof(int));
                cudaMemcpyToSymbol(d_camPos, camPos.ToArray(), sizeof(int));
                cudaMemcpyToSymbol(d_ViewProjectionMatrixInverse, viewProjectionInv.ToArray(), 16 * sizeof(float));
                CHECK_FOR_CUDA_ERROR();

                origin->Extend(rays);
                dir->Extend(rays);

                unsigned int blocks, threads;
                Calc1DKernelDimensions(rays, blocks, threads);
                CreateRays<<<blocks, threads>>>(origin->GetDeviceData(),
                                                dir->GetDeviceData());
                CHECK_FOR_CUDA_ERROR();
                if (visualizeRays){
                    RenderRayDir<<<blocks, threads>>>(dir->GetDeviceData(),
                                                      canvasData);
                    CHECK_FOR_CUDA_ERROR();
                    return;
                }

                Calc1DKernelDimensions(rays, blocks, threads, 128);
                int smemSize = threads * sizeof(float3) * 3;
                START_TIMER(timerID);
                logger.info << "BruteTracing<<<" << blocks << ", " << threads << ", " << smemSize << ">>>" << logger.end;
                BruteTracing<<<blocks, threads, smemSize>>>(origin->GetDeviceData(), dir->GetDeviceData(),
                                                            geom->GetP0Data(), geom->GetP1Data(), geom->GetP2Data(), 
                                                            geom->GetColor0Data(),
                                                            canvasData,
                                                            geom->GetSize());
                PRINT_TIMER(timerID, "Brute tracing");
                CHECK_FOR_CUDA_ERROR();
                
            }

        }
    }
}
