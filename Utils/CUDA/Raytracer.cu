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

#include <Utils/CUDA/Kernels/ColorKernels.h>

            RayTracer::RayTracer(TriangleMap* map)
                : map(map) {
                
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

            RayTracer::~RayTracer(){}

            __constant__ int d_rays;
            __constant__ int d_screenHeight;
            __constant__ int d_screenWidth;

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

            __host__ void TraceNodeHost(float3 origin, float3 direction, 
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

                logger.info << "tMin = " << tMin << logger.end;
                float tSplit = (splitPos - ori) / dir;
                logger.info << "tSplit = (" << splitPos << " - " << ori << ") / " << dir << " = " << tSplit << logger.end;
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
                                      float4 *n0s, float4 *n1s, float4 *n2s,
                                      uchar4 *c0s,
                                      uchar4 *canvas){
                
                const int id = blockDim.x * blockIdx.x + threadIdx.x;
                
                if (id < d_rays){
                    
                    float3 origin = make_float3(origins[id]);
                    float3 direction = make_float3(directions[id]);

                    float3 tHit;
                    tHit.x = 0.0f;

                    float4 color = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

                    do {
                        float tNext = fInfinity;
                        int node = 0;
                        char info = nodeInfo[node];
                        
                        while((info & 3) != KDNode::LEAF){
                            // Trace
                            float splitValue = splitPos[node];
                            int left = leftChild[node];
                            int right = rightChild[node];

                            TraceNode(origin, direction, info & 3, splitValue, left, right, tHit.x,
                                      node, tNext);
                                                        
                            info = nodeInfo[node];
                            
                            if (info & KDNode::PROXY){
                                node = leftChild[node];
                                info = nodeInfo[node];
                            }
                        }

                        tHit.x = tNext;

                        int2 primInfo = primitiveInfo[node];
                        int primHit = -1;
                        int triangles = primInfo.y;
                        while (triangles){
                            int i = __ffs(triangles) - 1;
                            
                            int prim = primIndices[primInfo.x + i];

                            float3 hitCoords;
                            bool hit = TriangleRayIntersection(make_float3(v0[prim]), make_float3(v1[prim]), make_float3(v2[prim]), 
                                                               origin, direction, hitCoords);

                            if (hit && hitCoords.x < tHit.x){
                                primHit = prim;
                                tHit = hitCoords;
                            }
                            
                            triangles -= 1<<i;
                        }

                        if (primHit != -1){
                            float4 newColor = Lighting(tHit, origin, direction, 
                                                       n0s[primHit], n1s[primHit], n2s[primHit],
                                                       c0s[primHit]);
                            
                            color = BlendColor(color, newColor);
                        }

                    } while(tHit.x < fInfinity && color.w < 0.97f);

                    canvas[id] = make_uchar4(color.x * 255, color.y * 255, color.z * 255, color.w * 255);
                }
            }

            void RayTracer::Trace(IRenderCanvas* canvas, uchar4* canvasData){
                //logger.info << "Trace!" << logger.end;

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

                TriangleNode* nodes = map->GetNodes();
                GeometryList* geom = map->GetGeometry();

                unsigned int blocks, threads;
                Calc1DKernelDimensions(rays, blocks, threads, 128);
                //START_TIMER(timerID); 
                KDRestart<<<blocks, threads>>>(origin->GetDeviceData(), direction->GetDeviceData(),
                                               nodes->GetInfoData(), nodes->GetSplitPositionData(),
                                               nodes->GetLeftData(), nodes->GetRightData(),
                                               nodes->GetPrimitiveInfoData(),
                                               map->GetPrimitiveIndices()->GetDeviceData(),
                                               geom->GetP0Data(), geom->GetP1Data(), geom->GetP2Data(),
                                               geom->GetNormal0Data(), geom->GetNormal1Data(), geom->GetNormal2Data(),
                                               geom->GetColor0Data(),
                                               canvasData);
                //PRINT_TIMER(timerID, "KDRestart");
                CHECK_FOR_CUDA_ERROR();
                                               
            }

            void RayTracer::HostTrace(float3 origin, float3 direction, Scene::TriangleNode* nodes){

                logger.info << "Origin " << Convert::ToString(origin) << logger.end;
                logger.info << "Direction " << Convert::ToString(direction) << logger.end;

                float3 tHit;
                tHit.x = 0.0f;

                float4 color = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

                do {
                    float tNext = fInfinity;
                    int node = 0;
                    char info;
                    cudaMemcpy(&info, nodes->GetInfoData() + node, sizeof(char), cudaMemcpyDeviceToHost);
                    CHECK_FOR_CUDA_ERROR();

                    while ((info & 3) != KDNode::LEAF){
                        logger.info << "Tracing " << node << " with info " << (int)info << logger.end;
                        
                        float splitValue;
                        cudaMemcpy(&splitValue, nodes->GetSplitPositionData() + node, sizeof(float), cudaMemcpyDeviceToHost);
                        CHECK_FOR_CUDA_ERROR();

                        int left;
                        cudaMemcpy(&left, nodes->GetLeftData() + node, sizeof(int), cudaMemcpyDeviceToHost);
                        int right;
                        cudaMemcpy(&right, nodes->GetRightData() + node, sizeof(int), cudaMemcpyDeviceToHost);
                        CHECK_FOR_CUDA_ERROR();
                        
                        TraceNode(origin, direction, info & 3, splitValue, left, right, tHit.x,
                                  node, tNext);

                        //logger.info << "tNext " << tNext << logger.end;

                        cudaMemcpy(&info, nodes->GetInfoData() + node, sizeof(char), cudaMemcpyDeviceToHost);

                        if (info & KDNode::PROXY){
                            cudaMemcpy(&node, nodes->GetLeftData() + node, sizeof(int), cudaMemcpyDeviceToHost);
                            cudaMemcpy(&info, nodes->GetInfoData() + node, sizeof(char), cudaMemcpyDeviceToHost);
                        }
                    }

                    logger.info << "Found leaf: " << node << "\n" << logger.end;
                    
                    tHit.x = tNext;
                    
                    int2 primInfo;
                    cudaMemcpy(&primInfo, nodes->GetPrimitiveInfoData() + node, sizeof(int2), cudaMemcpyDeviceToHost);
                    CHECK_FOR_CUDA_ERROR();
                    int primHit = -1;
                    int triangles = primInfo.y;
                    while (triangles){
                        int i = ffs(triangles) - 1;

                        int prim;
                        cudaMemcpy(&prim, map->GetPrimitiveIndices()->GetDeviceData() + primInfo.x + i, sizeof(int), cudaMemcpyDeviceToHost);
                        CHECK_FOR_CUDA_ERROR();

                        triangles -= 1<<i;
                    }

                    

                } while(tHit.x < fInfinity && color.w < 0.97f);
            }

        }
    }
}
