// Raytracer interface for CUDA
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Utils/CUDA/IRayTracer.h>

#include <Display/IRenderCanvas.h>
#include <Display/IViewingVolume.h>
#include <Math/Matrix.h>
#include <Utils/CUDA/Convert.h>
#include <Utils/CUDA/Utils.h>

using namespace OpenEngine::Display;
using namespace OpenEngine::Math;
using namespace OpenEngine::Resources::CUDA;

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {

            __constant__ float3 d_camPos;
            __constant__ int d_rays;
            __constant__ int d_screenHeight;
            __constant__ int d_screenWidth;
            __constant__ float d_ViewProjectionMatrixInverse[16];

            IRayTracer::IRayTracer() 
                : visualizeRays(false), intersectionAlgorithm(WOOP), 
                  leafSkipping(true), printTiming(false) {

                origin = new CUDADataBlock<1, float4>(1);
                direction = new CUDADataBlock<1, float4>(1);
            }
            
            IRayTracer::~IRayTracer() {
                if (origin) delete origin;
                if (direction) delete direction;
            }
            
            __device__ __host__ float3 Unproject(float2 screenPos, float* vpMatInv, float3 origin){
                float4 currentPos = make_float4(vpMatInv[0] * screenPos.x + vpMatInv[4] * screenPos.y + vpMatInv[8] + vpMatInv[12],
                                                vpMatInv[1] * screenPos.x + vpMatInv[5] * screenPos.y + vpMatInv[9] + vpMatInv[13],
                                                vpMatInv[2] * screenPos.x + vpMatInv[6] * screenPos.y + vpMatInv[10] + vpMatInv[14],
                                                vpMatInv[3] * screenPos.x + vpMatInv[7] * screenPos.y + vpMatInv[11] + vpMatInv[15]);
                
                float3 rayEnd = make_float3(currentPos.x / currentPos.w, currentPos.y / currentPos.w, currentPos.z / currentPos.w);
                return normalize(rayEnd - origin);
            }
            
            __global__ void CreateRays(float4* origin,
                                       float4* dir){
                const int id = blockDim.x * blockIdx.x + threadIdx.x;
                
                if (id < d_rays){
                    origin[id] = make_float4(d_camPos, 1.0f);

                    int x = id % d_screenWidth;
                    int y = id / d_screenWidth;
                    float2 screenPos = make_float2((x / (float)d_screenWidth) * 2.0f - 1.0f,
                                                   (y / (float)d_screenHeight) * 2.0f - 1.0f);
                    
                    float3 rayDir = Unproject(screenPos, d_ViewProjectionMatrixInverse, d_camPos);
                    dir[id] = make_float4(rayDir, 0.0f);
                }
            }

            void IRayTracer::CreateInitialRays(IRenderCanvas* canvas){
                Matrix<4, 4, float> viewProjection = 
                    canvas->GetViewingVolume()->GetViewMatrix() *
                    canvas->GetViewingVolume()->GetProjectionMatrix();
                
                Matrix<4, 4, float> viewProjectionInv = viewProjection.GetInverse();
                Vector<3, float> camPos = canvas->GetViewingVolume()->GetPosition();
                float3 h_camPos; h_camPos.x = camPos.Get(0); h_camPos.y = camPos.Get(1); h_camPos.z = camPos.Get(2);
                
                int height = canvas->GetHeight();
                screenWidth = canvas->GetWidth();
                
                int rays = height * screenWidth;
                cudaMemcpyToSymbol(d_screenWidth, &screenWidth, sizeof(int));
                cudaMemcpyToSymbol(d_screenHeight, &height, sizeof(int));
                cudaMemcpyToSymbol(d_rays, &rays, sizeof(int));
                cudaMemcpyToSymbol(d_camPos, &h_camPos, sizeof(float3));
                cudaMemcpyToSymbol(d_ViewProjectionMatrixInverse, viewProjectionInv.ToArray(), 16 * sizeof(float));
                CHECK_FOR_CUDA_ERROR();

                origin->Extend(rays);
                direction->Extend(rays);
                
                unsigned int blocks, threads;
                Calc1DKernelDimensions(rays, blocks, threads);
                CreateRays<<<blocks, threads>>>(origin->GetDeviceData(),
                                                direction->GetDeviceData());
                CHECK_FOR_CUDA_ERROR();
            }
            
            __global__ void RenderRayDir(float4* dir, uchar4 *canvas){
                const int id = blockDim.x * blockIdx.x + threadIdx.x;
                
                if (id < d_rays){
                    float4 d = dir[id] * 0.5f + make_float4(0.5f);
                    canvas[id] = make_uchar4(d.x * 255, d.y * 255, d.z * 255, d.w * 255);
                }
            }

            void IRayTracer::RenderRays(uchar4 *canvas, int rays){
                unsigned int blocks, threads;
                Calc1DKernelDimensions(rays, blocks, threads, 128);
                RenderRayDir<<<blocks, threads>>>(direction->GetDeviceData(),
                                                  canvas);
                CHECK_FOR_CUDA_ERROR();

            }
            
        }
    }
}
