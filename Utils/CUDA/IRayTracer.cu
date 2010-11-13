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
#include <Utils/CUDA/Utils.h>

using namespace OpenEngine::Display;
using namespace OpenEngine::Math;
using namespace OpenEngine::Resources::CUDA;

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {
            
            IRayTracer::IRayTracer() 
                : visualizeRays(false) {

                origin = new CUDADataBlock<1, float4>(1);
                dir = new CUDADataBlock<1, float4>(1);
            }
            
            IRayTracer::~IRayTracer() {
                if (origin) delete origin;
                if (dir) delete dir;
            }

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
                return normalize(rayEnd - origin);
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

            void IRayTracer::CreateInitialRays(IRenderCanvas* canvas){
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
            }
            
        }
    }
}
