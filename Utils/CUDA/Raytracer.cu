// Raytracer class for CUDA
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Meta/OpenGL.h>
#include <Utils/CUDA/Raytracer.h>
#include <Utils/CUDA/Convert.h>
#include <Utils/CUDA/Utils.h>

namespace OpenEngine {
    using namespace Display;
    using namespace Resources;
    using namespace Resources::CUDA;
    namespace Utils {
        namespace CUDA {

            RayTracer::RayTracer(TriangleMap* map)
                : map(map) {
                origin = new CUDADataBlock<1, float4>(1);
                dir = new CUDADataBlock<1, float4>(1);
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
                return normalize(rayEnd-origin);
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
                    
                    //dir[id] = make_float4(x, y, screenPos.x, screenPos.y);

                    float3 rayDir = Unproject(screenPos, d_ViewProjectionMatrixInverse, d_camPos);
                    dir[id] = make_float4(rayDir, 0.0f);
                }
            }

            void RayTracer::Trace(IRenderCanvas* canvas){
                //logger.info << "Trace!" << logger.end;

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

                /*
                logger.info << "dirs: " << Convert::ToString(dir->GetDeviceData(), 5) << logger.end;

                float2 screenPos = make_float2(-1.0f, -1.0f);
                float3 origin = make_float3(camPos.Get(0), camPos.Get(1), camPos.Get(2));
                float3 rayDir = Unproject(screenPos, viewProjectionInv.ToArray(), origin);
                logger.info << "RayDir for upper left cornor: " << Convert::ToString(rayDir) << logger.end;
                */

                GLint fb, tex;
                glGetIntegerv(GL_FRAMEBUFFER_BINDING_EXT, &fb);
                CHECK_FOR_GL_ERROR();

                glGetFramebufferAttachmentParameterivEXT(GL_FRAMEBUFFER_EXT, 
                                                         GL_COLOR_ATTACHMENT0_EXT,
                                                         GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME_EXT, 
                                                         &tex);
                CHECK_FOR_GL_ERROR();

                //logger.info << "Frame buffer " << fb << logger.end;
                //logger.info << "Texture " << tex << logger.end;

                glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
                glBindTexture(GL_TEXTURE_2D, tex);
                CHECK_FOR_GL_ERROR();
                
                cudaGraphicsResource *backBuffer;
                //cudaGraphicsGLRegisterImage(&backBuffer, 1, GL_RENDERBUFFER, cudaGraphicsMapFlagsWriteDiscard);
                //cudaGraphicsGLRegisterImage(&backBuffer, tex, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard);
                cudaGraphicsGLRegisterImage(&backBuffer, tex, GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly);
                CHECK_FOR_CUDA_ERROR();

                cudaStream_t stream;
                cudaStreamCreate(&stream);
                //logger.info << "stream " << stream << logger.end;
                //cudaGraphicsMapResources(1, &backBuffer, stream);
                cudaGraphicsMapResources(1, &backBuffer, 0);
                CHECK_FOR_CUDA_ERROR();
                
                size_t bytes;
                uchar4* pixels;
                cudaGraphicsResourceGetMappedPointer((void**)&pixels, &bytes,
                                                     backBuffer);
                CHECK_FOR_CUDA_ERROR();

                logger.info << "size: " << bytes / sizeof(uchar4) << logger.end;

                cudaGraphicsUnmapResources(1, &backBuffer, 0);
                CHECK_FOR_CUDA_ERROR();

                cudaGraphicsUnregisterResource(backBuffer);
                CHECK_FOR_CUDA_ERROR();
            }

        }
    }
}
