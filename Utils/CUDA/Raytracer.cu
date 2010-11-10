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

            __global__ void RenderRayDir(float4* dir, uchar4 *canvas){
                const int id = blockDim.x * blockIdx.x + threadIdx.x;
                
                if (id < d_rays){
                    float4 d = dir[id] * 0.5f + make_float4(0.5f);
                    canvas[id] = make_uchar4(d.x * 255, d.y * 255, d.z * 255, d.w * 255);
                }
            }

            __global__ void KDRestart(float4* origins, float4* directions,
                                      char* nodeInfo, float* splitPos,
                                      int* leftChild, int* rightChild,
                                      uchar4 *canvas){
                
                const int id = blockDim.x * blockIdx.x + threadIdx.x;
                
                if (id < d_rays){
                    
                    float4 origin = origins[id];
                    float4 direction = directions[id];

                    float tMin = 0.0f;
                    while (tMin != fInfinity){
                        float tNext = fInfinity;
                        int node = 0;
                        char info = nodeInfo[node];
                        
                        while(!(info &= KDNode::PROXY)){
                            // Trace
                            float ori, dir;
                            switch(info & 3){
                            case KDNode::X:
                                ori = origin.x; dir = direction.x;
                                break;
                            case KDNode::Y:
                                ori = origin.y; dir = direction.y;
                                break;
                            case KDNode::Z:
                                ori = origin.z; dir = direction.z;
                                break;
                            }
                            
                            float tSplit = (splitPos[id] - ori) / dir;
                            int left = leftChild[id];
                            int right = rightChild[id];
                            int lowerChild = 0 < dir ? left : right;
                            int upperChild = 0 < dir ? right : left;
                            if (tMin < tSplit){
                                node = lowerChild;
                                tNext = min(tSplit, tNext);
                            }else
                                node = upperChild;
                            
                        }

                        tMin = tNext;

                        // Test intersection
                        

                        // Debug
                        float4 pos = origin + tNext * direction;
                        canvas[id] = make_uchar4(pos.x * 51, pos.y * 51, pos.z * 51, 255);
                        tMin = fInfinity;
                    }
                }
            }

            void RayTracer::Trace(IRenderCanvas* canvas, uchar4* canvasData){
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

                if (visualizeRays){
                    RenderRayDir<<<blocks, threads>>>(dir->GetDeviceData(),
                                                      canvasData);
                    CHECK_FOR_CUDA_ERROR();
                    return;
                }
                
                KDRestart<<<blocks, threads>>>(origin->GetDeviceData(), dir->GetDeviceData(),
                                               map->nodes->GetInfoData(), map->nodes->GetSplitPositionData(),
                                               map->nodes->GetLeftData(), map->nodes->GetRightData(),
                                               canvasData);
                CHECK_FOR_CUDA_ERROR();
                                               
                /*
                logger.info << "dirs: " << Convert::ToString(dir->GetDeviceData(), 5) << logger.end;

                float2 screenPos = make_float2(-1.0f, -1.0f);
                float3 origin = make_float3(camPos.Get(0), camPos.Get(1), camPos.Get(2));
                float3 rayDir = Unproject(screenPos, viewProjectionInv.ToArray(), origin);
                logger.info << "RayDir for upper left cornor: " << Convert::ToString(rayDir) << logger.end;
                */

            }

        }
    }
}
