// Raytracer class for CUDA
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Utils/CUDA/Raytracer.h>

#include <Display/IViewingVolume.h>
#include <Display/IRenderCanvas.h>
#include <Scene/TriangleNode.h>
#include <Utils/CUDA/Convert.h>
#include <Utils/CUDA/TriangleMap.h>
#include <Utils/CUDA/Utils.h>

#define MAX_THREADS 128

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

            RayTracer::~RayTracer() {}

            __constant__ int d_rays;

            /**
             * http://www.sven-woop.de/publications/Diplom_SvenWoop_Final.pdf

             * Not working yet. Precalc inverted matrix in
             * geometrylist. Then do actual intersection in
             * raytracer. Perhaps calc uv first and provide early out.
             */
            inline __host__ __device__ bool WoopIntersection(float3 a, float3 b, float3 c,
                                                             float3 origin, float3 direction,
                                                             float3 &hit){
                const float3 m_x = a - c;
                const float3 m_y = b - c;
                const float3 m_z = make_float3(1.0f, 0.0f, 0.0f) - m_x - m_y;

                const float3 n = c;

                origin = make_float3(dot(m_x, origin), 
                                     dot(m_y, origin), 
                                     dot(m_z, origin)) + n;
                direction = make_float3(dot(m_x, direction), 
                                        dot(m_y, direction), 
                                        dot(m_z, direction));

                hit.x = -origin.z / direction.z;
                hit.y = hit.x * direction.x + origin.x;
                hit.z = hit.x * direction.y + origin.y;

                return hit.x >= 0.0f && hit.y >= 0.0f && hit.z >= 0.0f && hit.y + hit.z <= 1.0f;
            }


            __device__ __host__ void TraceNode(float3 origin, float3 direction, 
                                               char axis, float splitPos,
                                               int left, int right, float tMin,
                                               int &node, float &tNext){
                float ori, dir;
                switch(axis){
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

                if (tMin < tSplit){
                    node = 0 < dir ? left : right;
                    tNext = min(tSplit, tNext);
                }else
                    node = 0 < dir ? right : left;
            }

            __global__ void 
            __launch_bounds__(MAX_THREADS, 2) 
                KDRestart(float4* origins, float4* directions,
                          char* nodeInfo, float* splitPos,
                          int2* children,
                          int* nodePrimIndex, KDNode::bitmap* primBitmap,
                          int *primIndices, 
                          float4 *v0, float4 *v1, float4 *v2,
                          float4 *n0s, float4 *n1s, float4 *n2s,
                          uchar4 *c0s,
                          uchar4 *canvas,
                          int screenWidth){
                
                int id = blockDim.x * blockIdx.x + threadIdx.x;
                
                if (id < d_rays){
                
                    id = IRayTracer::PacketIndex(id, screenWidth);
    
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
                            int2 childPair = children[node];

                            TraceNode(origin, direction, info & 3, splitValue, childPair.x, childPair.y, tHit.x,
                                      node, tNext);
                                                        
                            info = nodeInfo[node];
                        }

                        tHit.x = tNext;

                        int primIndex = nodePrimIndex[node];
                        KDNode::bitmap triangles = primBitmap[node];
                        int primHit = -1;
                        while (triangles){
                            int i = firstBitSet(triangles) - 1;
                            
                            int prim = primIndices[primIndex + i];

                            float3 hitCoords;
                            bool hit = TriangleRayIntersection(make_float3(v0[prim]), make_float3(v1[prim]), make_float3(v2[prim]), 
                                                               origin, direction, hitCoords);

                            if (hit && hitCoords.x < tHit.x){
                                primHit = prim;
                                tHit = hitCoords;
                            }
                            
                            triangles -= KDNode::bitmap(1)<<i;
                        }

                        if (primHit != -1){
                            float4 newColor = Lighting(tHit, origin, direction, 
                                                       n0s[primHit], n1s[primHit], n2s[primHit],
                                                       c0s[primHit]);

                            color = BlendColor(color, newColor);

                            tHit.x = 0.0f;
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

                cudaMemcpyToSymbol(d_rays, &rays, sizeof(int));

                if (visualizeRays){
                    RenderRays(canvasData, rays);
                    return;
                }

                TriangleNode* nodes = map->GetNodes();
                GeometryList* geom = map->GetGeometry();
                
                unsigned int blocks, threads;
                Calc1DKernelDimensions(rays, blocks, threads, MAX_THREADS);
                START_TIMER(timerID);
                KDRestart<<<blocks, threads>>>(origin->GetDeviceData(), direction->GetDeviceData(),
                                               nodes->GetInfoData(), nodes->GetSplitPositionData(),
                                               nodes->GetChildrenData(),
                                               nodes->GetPrimitiveIndexData(),
                                               nodes->GetPrimitiveBitmapData(),
                                               map->GetPrimitiveIndices()->GetDeviceData(),
                                               geom->GetP0Data(), geom->GetP1Data(), geom->GetP2Data(),
                                               geom->GetNormal0Data(), geom->GetNormal1Data(), geom->GetNormal2Data(),
                                               geom->GetColor0Data(),
                                               canvasData,
                                               width);
                PRINT_TIMER(timerID, "KDRestart");
                CHECK_FOR_CUDA_ERROR();                                               
            }

            void RayTracer::HostTrace(float3 origin, float3 direction, TriangleNode* nodes){

                GeometryList* geom = map->GetGeometry();

                float3 tHit;
                tHit.x = 0.0f;

                float4 color = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

                do {
                    logger.info << "=== Ray:  " << Convert::ToString(origin) << " -> " << Convert::ToString(direction) << " ===\n" << logger.end;
                
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

                        int2 children;
                        cudaMemcpy(&children, nodes->GetChildrenData() + node, sizeof(int2), cudaMemcpyDeviceToHost);
                        CHECK_FOR_CUDA_ERROR();
                        
                        TraceNode(origin, direction, info & 3, splitValue, children.x, children.y, tHit.x,
                                  node, tNext);

                        //logger.info << "tNext " << tNext << logger.end;

                        cudaMemcpy(&info, nodes->GetInfoData() + node, sizeof(char), cudaMemcpyDeviceToHost);
                    }

                    logger.info << "Found leaf: " << node << "\n" << logger.end;
                    
                    tHit.x = tNext;
                    
                    int primIndex;
                    cudaMemcpy(&primIndex, nodes->GetPrimitiveIndexData() + node, sizeof(int), cudaMemcpyDeviceToHost);
                    CHECK_FOR_CUDA_ERROR();
                    int primHit = -1;
                    KDNode::bitmap triangles;
                    cudaMemcpy(&triangles, nodes->GetPrimitiveBitmapData() + node, sizeof(KDNode::bitmap), cudaMemcpyDeviceToHost);
                    while (triangles){
                        int i = ffs(triangles) - 1;

                        //logger.info << "Testing indice " << primInfo.x << " + " << i << " = " << primInfo.x + i << logger.end;

                        int prim;
                        cudaMemcpy(&prim, map->GetPrimitiveIndices()->GetDeviceData() + primIndex + i, sizeof(int), cudaMemcpyDeviceToHost);
                        CHECK_FOR_CUDA_ERROR();
                        
                        //logger.info << "Testing primitive " << prim << logger.end;

                        float3 v0, v1, v2;
                        cudaMemcpy(&v0, geom->GetP0Data() + prim, sizeof(float3), cudaMemcpyDeviceToHost);
                        cudaMemcpy(&v1, geom->GetP1Data() + prim, sizeof(float3), cudaMemcpyDeviceToHost);
                        cudaMemcpy(&v2, geom->GetP2Data() + prim, sizeof(float3), cudaMemcpyDeviceToHost);
                        CHECK_FOR_CUDA_ERROR();

                        float3 hitCoords;
                        bool hit = TriangleRayIntersection(v0, v1, v2, 
                                                           origin, direction, hitCoords);

                        if (hit && hitCoords.x < tHit.x){
                            primHit = prim;
                            tHit = hitCoords;
                        }
                        
                        triangles -= KDNode::bitmap(1)<<i;
                    }
                    
                    //logger.info << "\n" << logger.end;
                    
                    if (primHit != -1){
                        float4 n0, n1, n2;
                        cudaMemcpy(&n0, geom->GetNormal0Data() + primHit, sizeof(float4), cudaMemcpyDeviceToHost);
                        cudaMemcpy(&n1, geom->GetNormal1Data() + primHit, sizeof(float4), cudaMemcpyDeviceToHost);
                        cudaMemcpy(&n2, geom->GetNormal2Data() + primHit, sizeof(float4), cudaMemcpyDeviceToHost);
                        CHECK_FOR_CUDA_ERROR();

                        uchar4 c0;
                        cudaMemcpy(&c0, geom->GetColor0Data() + primHit, sizeof(uchar4), cudaMemcpyDeviceToHost);                        

                        logger.info << "Prim color: " << Convert::ToString(c0) << logger.end;

                        float4 newColor = Lighting(tHit, origin, direction, 
                                                   n0, n1, n2,
                                                   c0);

                        logger.info << "New color: " << Convert::ToString(newColor) << logger.end;
                        
                        color = BlendColor(color, newColor);

                        logger.info << "Color: " << Convert::ToString(color) << "\n" << logger.end;

                        tHit.x = 0.0f;
                    }

                } while(tHit.x < fInfinity && color.w < 0.97f);

                logger.info << "Final color: " << Convert::ToString(color) << logger.end;
            }

        }
    }
}
