// Raytracer class for CUDA
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Meta/OpenGL.h>
#include <Display/IViewingVolume.h>
#include <Scene/TriangleNode.h>
#include <Utils/CUDA/Raytracer.h>
#include <Utils/CUDA/Convert.h>
#include <Utils/CUDA/Utils.h>

namespace OpenEngine {
    using namespace Display;
    using namespace Resources;
    using namespace Resources::CUDA;
    using namespace Scene;
    namespace Utils {
        namespace CUDA {

            RayTracer::RayTracer(TriangleMap* map)
                : map(map) {
                
                cutCreateTimer(&timerID);

                origin = new CUDADataBlock<1, float4>(1);
                dir = new CUDADataBlock<1, float4>(1);
            }

            RayTracer::~RayTracer(){
                delete origin;
                delete dir;
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

            __device__ __host__ void TraceNode(float3 origin, float3 direction, 
                                               char axes, float splitPos,
                                               int left, int right, float tMin,
                                               int &node, float &tNext){
                float ori, dir;
                switch(axes){
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

                float tSplit = (splitPos - ori) / dir;
                int lowerChild = 0 < dir ? left : right;
                int upperChild = 0 < dir ? right : left;
                if (tMin < tSplit){
                    node = lowerChild;
                    tNext = min(tSplit, tNext);
                }else
                    node = upperChild;
            }

            __global__ void KDRestart(float4* origins, float4* directions,
                                      char* nodeInfo, float* splitPos,
                                      int* leftChild, int* rightChild,
                                      int2 *primitiveInfo, 
                                      int *primIndices, 
                                      float4 *v0, float4 *v1, float4 *v2,
                                      uchar4 *c0,
                                      uchar4 *canvas){
                
                const int id = blockDim.x * blockIdx.x + threadIdx.x;
                
                if (id < d_rays){
                    
                    float3 origin = make_float3(origins[id]);
                    float3 direction = make_float3(directions[id]);

                    uchar4 color = make_uchar4(0, 0, 0, 0);
                    
                    // @OPT early out bounding box maximums

                    float tMin = 0.0f;
                    while (tMin != fInfinity){
                        float tNext = fInfinity;
                        int node = 0;
                        char info = nodeInfo[node];
                        
                        while((info & 3) != KDNode::LEAF){
                            // Trace
                            float splitValue = splitPos[node];
                            int left = leftChild[node];
                            int right = rightChild[node];

                            TraceNode(origin, direction, info & 3, splitValue, left, right, tMin,
                                      node, tNext);
                                                        
                            info = nodeInfo[node];
                            
                            if (info & KDNode::PROXY)
                                node = leftChild[node];
                        }

                        tMin = tNext;

                        // Test intersection
                        int2 primInfo = primitiveInfo[node];
                        int triangles = primInfo.y;
                        while (triangles){
                            int i = __ffs(triangles) - 1;

                            int prim = primIndices[primInfo.x + i];
                            
                            float3 hitCoords;
                            bool hit = TriangleRayIntersection(make_float3(v0[prim]), make_float3(v1[prim]), make_float3(v2[prim]), 
                                                               origin, direction, hitCoords);

                            if (hit){
                                color = c0[prim];
                                tMin = fInfinity;
                            }

                            triangles -= 1<<i;
                        }
                    }

                    canvas[id] = make_uchar4(color.x, color.y, color.z, 255);
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

                /*
                float3 o = make_float3(0.0f, 0.0f, 0.0f);
                float3 d = make_float3(1.0f, 1.0f, 1.0f);
                HostTrace(o, d, map->nodes);
                */

                if (visualizeRays){
                    RenderRayDir<<<blocks, threads>>>(dir->GetDeviceData(),
                                                      canvasData);
                    CHECK_FOR_CUDA_ERROR();
                    return;
                }

                TriangleNode* nodes = map->GetNodes();
                GeometryList* geom = map->GetGeometry();

                START_TIMER(timerID); 
                KDRestart<<<blocks, threads>>>(origin->GetDeviceData(), dir->GetDeviceData(),
                                               nodes->GetInfoData(), nodes->GetSplitPositionData(),
                                               nodes->GetLeftData(), nodes->GetRightData(),
                                               nodes->GetPrimitiveInfoData(),
                                               map->GetPrimitiveIndices()->GetDeviceData(),
                                               geom->GetP0Data(), geom->GetP1Data(), geom->GetP2Data(),
                                               geom->GetColor0Data(),
                                               canvasData);
                PRINT_TIMER(timerID, "KDRestart");
                CHECK_FOR_CUDA_ERROR();
                                               
            }

            void RayTracer::HostTrace(float3 origin, float3 direction, Scene::TriangleNode* nodes){
                
                float tMin = 0.0f;
                while (tMin != fInfinity){
                    float tNext = fInfinity;
                    int node = 0;
                    char info;
                    cudaMemcpy(&info, nodes->GetInfoData() + node, sizeof(char), cudaMemcpyDeviceToHost);
                    CHECK_FOR_CUDA_ERROR();
                    
                    while (!(info & KDNode::PROXY) && (info & 3) != KDNode::LEAF){
                        logger.info << "Tracing " << node << logger.end;

                        float splitValue;
                        cudaMemcpy(&splitValue, nodes->GetSplitPositionData() + node, sizeof(float), cudaMemcpyDeviceToHost);
                        CHECK_FOR_CUDA_ERROR();
                        
                        int left;
                        cudaMemcpy(&left, nodes->GetLeftData() + node, sizeof(int), cudaMemcpyDeviceToHost);
                        //logger.info << "left child " << left << logger.end;
                        int right;
                        cudaMemcpy(&right, nodes->GetRightData() + node, sizeof(int), cudaMemcpyDeviceToHost);
                        //logger.info << "right child " << right << logger.end;
                        CHECK_FOR_CUDA_ERROR();
                        
                        TraceNode(origin, direction, info & 3, splitValue, left, right, tMin,
                                  node, tNext);

                        cudaMemcpy(&info, nodes->GetInfoData() + node, sizeof(char), cudaMemcpyDeviceToHost);

                        if (info & KDNode::PROXY)
                            cudaMemcpy(&node, nodes->GetLeftData() + node, sizeof(int), cudaMemcpyDeviceToHost);
                        
                        //logger.info << "tNext " << tNext << logger.end;
                        
                    }

                    logger.info << "Found leaf " << nodes->ToString(node) << logger.end;
                    
                    tMin = tNext;
                    //logger.info << "new tMin " << tMin << logger.end;
                    
                }
            }

        }
    }
}
