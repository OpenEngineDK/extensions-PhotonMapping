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

#include <Utils/CUDA/LoggerExtensions.h>

#define MAX_THREADS 64
#define MIN_BLOCKS 4

namespace OpenEngine {
    using namespace Display;
    using namespace Resources;
    using namespace Resources::CUDA;
    using namespace Scene;
    namespace Utils {
        namespace CUDA {
            
            __constant__ int d_rays;

            namespace RaytracerKDns{
#include <Utils/CUDA/Kernels/ColorKernels.h>
            } using namespace RaytracerKDns;

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

            template <bool invDir>
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
                
#ifdef __CUDA_ARCH__
                const float tSplit = invDir ? (splitPos - ori) * dir : __fdividef(splitPos - ori, dir);
#else
                const float tSplit = invDir ? (splitPos - ori) * dir : (splitPos - ori) / dir;
#endif

                if (tMin < tSplit){
                    node = 0 < dir ? left : right;
                    tNext = min(tSplit, tNext);
                }else
                    node = 0 < dir ? right : left;
            }

            template <bool useWoop, bool invDir>
            inline __host__ __device__ 
            uchar4 KDRestart(int id, float4* origins, float4* directions,
                           char* nodeInfo, float* splitPos,
                           int2* children,
                           int* nodePrimIndex, KDNode::bitmap* primBitmap,
                           int *primIndices, 
                           float4 *v0, float4 *v1, float4 *v2,
                           float4 *n0s, float4 *n1s, float4 *n2s,
                           uchar4 *c0s){
                    
                float3 origin = make_float3(FetchDeviceData(origins, id));
                float3 direction = make_float3(FetchDeviceData(directions, id));
                
#ifndef __CUDA_ARCH__
                logger.info << "=== Ray:  " << origin << " -> " << direction << " ===\n" << logger.end;
#endif

                float3 tHit;
                tHit.x = 0.0f;
                
                float4 color = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

                do {
                    float tNext = fInfinity;
                    int node = 0;
                    char info = FetchDeviceData(nodeInfo, node);

                    if (invDir) direction = make_float3(1.0f, 1.0f, 1.0f) / direction;
                    
                    while((info & 3) != KDNode::LEAF){
                        // Trace
#ifndef __CUDA_ARCH__
                        logger.info << "Tracing " << node << " with info " << (int)info << logger.end;
#endif
                        
                        float splitValue = FetchDeviceData(splitPos, node);
                        int2 childPair = FetchDeviceData(children, node);
                        
                        TraceNode<invDir>(origin, direction, info & 3, 
                                          splitValue, childPair.x, childPair.y, 
                                          tHit.x, node, tNext);
                                                        
                        info = FetchDeviceData(nodeInfo, node);
                    }
#ifndef __CUDA_ARCH__
                    logger.info << "Found leaf: " << node << "\n" << logger.end;
#endif
                    if (invDir) direction = make_float3(1.0f, 1.0f, 1.0f) / direction;
                    tHit.x = tNext;

                    int primIndex = FetchDeviceData(nodePrimIndex, node);
                    KDNode::bitmap triangles = FetchDeviceData(primBitmap, node);
                    int primHit = -1;
                    while (triangles){
                        int i = firstBitSet(triangles) - 1;
                        int prim = FetchDeviceData(primIndices, primIndex + i);
                            
                        if (useWoop){
                            IRayTracer::Woop(v0, v1, v2, prim,
                                             origin, direction, primHit, tHit);
                        }else{
                            IRayTracer::MoellerTrumbore(v0, v1, v2, prim,
                                                        origin, direction, primHit, tHit);
                        }
                            
                        triangles -= KDNode::bitmap(1)<<i;
                    }

#ifndef __CUDA_ARCH__
                    logger.info << "THit: " << tHit << "\n" << logger.end;
#endif

                    if (primHit != -1){
                        float4 newColor = Lighting(tHit, origin, direction, 
                                                   FetchDeviceData(n0s, primHit), FetchDeviceData(n1s, primHit), FetchDeviceData(n2s, primHit),
                                                   FetchDeviceData(c0s, primHit));
                        
                        color = BlendColor(color, newColor);
                        
                        tHit.x = 0.0f;
                    }

                } while(tHit.x < fInfinity && color.w < 0.97f);
                
                return make_uchar4(color.x * 255, color.y * 255, color.z * 255, color.w * 255);
            }
            
            template <bool useWoop, bool invDir>
            __global__ void 
            __launch_bounds__(MAX_THREADS, MIN_BLOCKS) 
                KDRestartKernel(float4* origins, float4* directions,
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

                    uchar4 color = KDRestart<useWoop, invDir>(id, origins, directions, 
                                                              nodeInfo, splitPos, children,
                                                              nodePrimIndex, primBitmap, primIndices, 
                                                              v0, v1, v2, n0s, n1s, n2s, c0s);
                    
                    DumpDeviceData(color, canvas, id);
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

                if (this->intersectionAlgorithm == WOOP){
                    float4 *woop0, *woop1, *woop2;
                    geom->GetWoopValues(&woop0, &woop1, &woop2);

                    KernelConf conf = KernelConf1D(rays, MAX_THREADS);
                    START_TIMER(timerID);
                    KDRestartKernel<true, true><<<conf.blocks, conf.threads>>>
                        (origin->GetDeviceData(), direction->GetDeviceData(),
                         nodes->GetInfoData(), nodes->GetSplitPositionData(),
                         nodes->GetChildrenData(),
                         nodes->GetPrimitiveIndexData(),
                         nodes->GetPrimitiveBitmapData(),
                         map->GetPrimitiveIndices()->GetDeviceData(),
                         woop0, woop1, woop2,
                         geom->GetNormal0Data(), geom->GetNormal1Data(), geom->GetNormal2Data(),
                         geom->GetColor0Data(),
                         canvasData,
                         width);
                    PRINT_TIMER(timerID, "KDRestart with Woop intersection");

                }else{               
                    KernelConf conf = KernelConf1D(rays, MAX_THREADS);
                    START_TIMER(timerID);
                    KDRestartKernel<false, true><<<conf.blocks, conf.threads>>>
                        (origin->GetDeviceData(), direction->GetDeviceData(),
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
                    PRINT_TIMER(timerID, "KDRestart with Möller-Trumbore");
                }
                CHECK_FOR_CUDA_ERROR();
                
                //HostTrace(320, 240, nodes);
            }

            void RayTracer::HostTrace(int x, int y, TriangleNode* nodes){

                int id = x + y * screenWidth;
                
                //TriangleNode* nodes = map->GetNodes();
                GeometryList* geom = map->GetGeometry();

                float4 *woop0, *woop1, *woop2;
                geom->GetWoopValues(&woop0, &woop1, &woop2);

                uchar4 color = KDRestart<true, true>
                    (id, origin->GetDeviceData(), direction->GetDeviceData(),
                     nodes->GetInfoData(), nodes->GetSplitPositionData(),
                     nodes->GetChildrenData(),
                     nodes->GetPrimitiveIndexData(),
                     nodes->GetPrimitiveBitmapData(),
                     map->GetPrimitiveIndices()->GetDeviceData(),
                     woop0, woop1, woop2,
                     geom->GetNormal0Data(), geom->GetNormal1Data(), geom->GetNormal2Data(),
                     geom->GetColor0Data());

                logger.info << "Final color: " << make_int4(color.x, color.y, color.z, color.w) << logger.end;
            }

        }
    }
}
