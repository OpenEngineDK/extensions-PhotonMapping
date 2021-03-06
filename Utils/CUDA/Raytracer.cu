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

#define MAX_THREADS 128
#define MIN_BLOCKS 2

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
                                               char* nodeInfo, float* splitPos,
                                               int2* children, float &tMin,
                                               int &node){
                float tNext = fInfinity;
                node = 0;
                char axis = FetchGlobalData(nodeInfo, node);
                
                if (invDir) direction = make_float3(1.0f, 1.0f, 1.0f) / direction;

                while((axis & 3) != KDNode::LEAF){
                    // Trace                    
                    float splitValue = FetchGlobalData(splitPos, node);
                    int2 childPair = FetchGlobalData(children, node);

                    CUDALogger("Tracing " << node << " with info " << (int)axis << " and splitPos " << splitValue);
                    
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
                    const float tSplit = invDir ? (splitValue - ori) * dir : __fdividef(splitValue - ori, dir);
#else
                    const float tSplit = invDir ? (splitValue - ori) * dir : (splitValue - ori) / dir;
#endif
                    
                    if (tMin < tSplit){
                        node = 0 < dir ? childPair.x : childPair.y;
                        tNext = min(tSplit, tNext);
                    }else
                        node = 0 < dir ? childPair.y : childPair.x;

                    axis = FetchGlobalData(nodeInfo, node);
                }

                tMin = tNext * 1.00001f;

                CUDALogger("Found leaf: " << node << "\n");
            }

            template <bool useWoop, bool invDir>
            inline __host__ __device__
            float ShadowRay(const float3 point, const float3 lightPos,
                           char *nodeInfo, float* splitPos,
                           int* nodePrimIndex, KDNode::bitmap* primBitmap,
                           int2 *children,
                           int* primIndices,
                           float4* v0, float4* v1, float4* v2){
                
                float3 direction = point - lightPos;

                CUDALogger("=== Shadow Ray:  " << point << " -> " << direction << " ===\n");

                float tMin = 0.0f;
                do {
                    int node;
                    TraceNode<invDir>(lightPos, direction, nodeInfo, 
                                      splitPos, children, 
                                      tMin, node);

                    if (tMin >= 0.999f) return 1.0f;

                    float3 tHit;
                    tHit.x = tMin;

                    int primIndex = FetchGlobalData(nodePrimIndex, node);
                    KDNode::bitmap triangles = FetchGlobalData(primBitmap, node);
                    int primHit = -1;
                    while (triangles){
                        int i = firstBitSet(triangles) - 1;
                        int prim = FetchGlobalData(primIndices, primIndex + i);

                        CUDALogger("tHit " << tHit);
                        
                        if (useWoop){
                            IRayTracer::Woop(v0, v1, v2, prim,
                                             lightPos, direction, primHit, tHit);
                        }else{
                            IRayTracer::MoellerTrumbore(v0, v1, v2, prim,
                                                        lightPos, direction, primHit, tHit);
                        }
                        
                        triangles -= KDNode::bitmap(1)<<i;
                    }                    

                    if (primHit != -1){
                        CUDALogger("Shadow ray intersected " << primHit << " with tHit " << tHit);
                        return 0.0f;
                    }

                } while (tMin < 0.999f);

                CUDALogger("No shadow");

                return 1.0;
            }
                
            template <bool useWoop, bool invDir, bool rayBoxIntersect>
            inline __host__ __device__ 
            uchar4 KDRestart(int id, float4* origins, float4* directions,
                             char* nodeInfo, float* splitPos,
                             int2* children,
                             int* nodePrimIndex, KDNode::bitmap* primBitmap,
                             float4* nodeMin, float4* nodeMax,
                             int *primIndices, 
                             float4 *v0, float4 *v1, float4 *v2,
                             float4 *n0s, float4 *n1s, float4 *n2s,
                             uchar4 *c0s){

                float3 origin = make_float3(FetchGlobalData(origins, id));
                float3 direction = make_float3(FetchGlobalData(directions, id));
                IRayTracer::AdjustRayDirection(direction);

                CUDALogger("=== Ray:  " << origin << " -> " << direction << " ===\n");

                float3 tHit;
                tHit.x = 0.0f;
                
                float4 color = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

                do {
                    int node;
                    TraceNode<invDir>(origin, direction, nodeInfo, 
                                      splitPos, children, 
                                      tHit.x, node);
                    
                    if (!rayBoxIntersect || 
                        IRayTracer::RayBoxIntersection<true>(origin, make_float3(1.0f) / direction,
                                                             make_float3(FetchGlobalData(nodeMin, node)), 
                                                             make_float3(FetchGlobalData(nodeMax, node)))){
                        
                        int primIndex = FetchGlobalData(nodePrimIndex, node);
                        KDNode::bitmap triangles = FetchGlobalData(primBitmap, node);
                        int primHit = -1;
                        while (triangles){
                            int i = firstBitSet(triangles) - 1;
                            int prim = FetchGlobalData(primIndices, primIndex + i);
                            
                            if (useWoop){
                                IRayTracer::Woop(v0, v1, v2, prim,
                                             origin, direction, primHit, tHit);
                            }else{
                                IRayTracer::MoellerTrumbore(v0, v1, v2, prim,
                                                            origin, direction, primHit, tHit);
                            }
                            
                            triangles -= KDNode::bitmap(1)<<i;
                        }
                        
                        CUDALogger("THit: " << tHit << "\n");
                        
                        if (primHit != -1){
                            CUDALogger("Primary ray intersected " << primHit);
                            /*
                            const float3 point = origin + tHit.x * direction;
                            const float shadow = ShadowRay<useWoop, invDir>(point, FetchDeviceData(d_lightPosition),
                                                                            nodeInfo, splitPos, nodePrimIndex,
                                                                            primBitmap, children, primIndices,
                                                                            v0, v1, v2);
                            */
                            const float shadow = 1.0f;
                            
                            float4 newColor = Lighting(tHit, origin, direction, 
                                                       n0s, n1s, n2s,
                                                       c0s, primHit, shadow);
                            
                            color = BlendColor(color, newColor);
                            
                            tHit.x = 0.0f;
                            IRayTracer::AdjustRayDirection(direction);

                            // Fake cubemap hack
                            /*
                            if (direction.x < 0.0f)
                                tHit.x = (-4.9f - origin.x) / direction.x;
                            else
                                tHit.x = (4.9f - origin.x) / direction.x;

                            if (direction.y < 0.0f)
                                tHit.x = min(tHit.x, (-4.9f - origin.y) / direction.y);
                            else
                                tHit.x = min(tHit.x, (4.9f - origin.y) / direction.y);

                            if (direction.z < 0.0f)
                                tHit.x = min(tHit.x, (-4.9f - origin.z) / direction.z);
                            else
                                tHit.x = min(tHit.x, (4.9f - origin.z) / direction.z);
                            */
                        }
                    }

                } while(tHit.x < fInfinity && color.w < 0.97f);
                
                return make_uchar4(color.x * 255, color.y * 255, color.z * 255, color.w * 255);
            }
            
            template <bool useWoop, bool invDir, bool rayBoxIntersect>
            __global__ void 
            __launch_bounds__(MAX_THREADS, MIN_BLOCKS) 
                KDRestartKernel(float4* origins, float4* directions,
                                char* nodeInfo, float* splitPos,
                                int2* children,
                                int* nodePrimIndex, KDNode::bitmap* primBitmap,
                                float4* nodeMin, float4* nodeMax,
                                int *primIndices, 
                                float4 *v0, float4 *v1, float4 *v2,
                                float4 *n0s, float4 *n1s, float4 *n2s,
                                uchar4 *c0s,
                                uchar4 *canvas,
                                int screenWidth){

                int id = blockDim.x * blockIdx.x + threadIdx.x;
                
                if (id < d_rays){
                    
                    id = IRayTracer::PacketIndex(id, screenWidth);

                    uchar4 color = KDRestart<useWoop, invDir, rayBoxIntersect>
                        (id, origins, directions, nodeInfo, 
                         splitPos, children, nodePrimIndex, primBitmap, 
                         nodeMin, nodeMax, primIndices, 
                         v0, v1, v2, n0s, n1s, n2s, c0s);
                    
                    DumpGlobalData(color, canvas, id);
                }
            }

            void RayTracer::Trace(IRenderCanvas* canvas, uchar4* canvasData){
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
                
                if (map->GetPropagateBoundingBox() == false) leafSkipping = false;

                KernelConf conf = KernelConf1D(rays, MAX_THREADS);
                START_TIMER(timerID);
                if (this->intersectionAlgorithm == WOOP){
                    float4 *woop0, *woop1, *woop2;
                    geom->GetWoopValues(&woop0, &woop1, &woop2);

                    if (leafSkipping){
                        KDRestartKernel<true, true, true><<<conf.blocks, conf.threads>>>
                            (origin->GetDeviceData(), direction->GetDeviceData(),
                             nodes->GetInfoData(), nodes->GetSplitPositionData(),
                             nodes->GetChildrenData(),
                             nodes->GetPrimitiveIndexData(),
                             nodes->GetPrimitiveBitmapData(),
                             nodes->GetAabbMinData(), nodes->GetAabbMaxData(), 
                             map->GetPrimitiveIndices()->GetDeviceData(),
                             woop0, woop1, woop2,
                             geom->GetNormal0Data(), geom->GetNormal1Data(), geom->GetNormal2Data(),
                             geom->GetColor0Data(),
                             canvasData,
                             width);
                        if (printTiming) PRINT_TIMER(timerID, "KDRestart with Woop intersection and leaf skipping");
                    }else{
                        KDRestartKernel<true, true, false><<<conf.blocks, conf.threads>>>
                            (origin->GetDeviceData(), direction->GetDeviceData(),
                             nodes->GetInfoData(), nodes->GetSplitPositionData(),
                             nodes->GetChildrenData(),
                             nodes->GetPrimitiveIndexData(),
                             nodes->GetPrimitiveBitmapData(),
                             nodes->GetAabbMinData(), nodes->GetAabbMaxData(), 
                             map->GetPrimitiveIndices()->GetDeviceData(),
                             woop0, woop1, woop2,
                             geom->GetNormal0Data(), geom->GetNormal1Data(), geom->GetNormal2Data(),
                             geom->GetColor0Data(),
                             canvasData,
                             width);
                        if (printTiming) PRINT_TIMER(timerID, "KDRestart with Woop intersection");
                    }

                }else{
                    if (leafSkipping){
                        KDRestartKernel<false, true, true><<<conf.blocks, conf.threads>>>
                            (origin->GetDeviceData(), direction->GetDeviceData(),
                             nodes->GetInfoData(), nodes->GetSplitPositionData(),
                             nodes->GetChildrenData(),
                             nodes->GetPrimitiveIndexData(),
                             nodes->GetPrimitiveBitmapData(),
                             nodes->GetAabbMinData(), nodes->GetAabbMaxData(), 
                             map->GetPrimitiveIndices()->GetDeviceData(),
                             geom->GetP0Data(), geom->GetP1Data(), geom->GetP2Data(),
                             geom->GetNormal0Data(), geom->GetNormal1Data(), geom->GetNormal2Data(),
                             geom->GetColor0Data(),
                             canvasData,
                             width);
                        if (printTiming) PRINT_TIMER(timerID, "KDRestart with Möller-Trumbore and leaf skipping");
                    }else{
                        KDRestartKernel<false, true, false><<<conf.blocks, conf.threads>>>
                            (origin->GetDeviceData(), direction->GetDeviceData(),
                             nodes->GetInfoData(), nodes->GetSplitPositionData(),
                             nodes->GetChildrenData(),
                             nodes->GetPrimitiveIndexData(),
                             nodes->GetPrimitiveBitmapData(),
                             nodes->GetAabbMinData(), nodes->GetAabbMaxData(), 
                             map->GetPrimitiveIndices()->GetDeviceData(),
                             geom->GetP0Data(), geom->GetP1Data(), geom->GetP2Data(),
                             geom->GetNormal0Data(), geom->GetNormal1Data(), geom->GetNormal2Data(),
                             geom->GetColor0Data(),
                             canvasData,
                             width);
                        if (printTiming) PRINT_TIMER(timerID, "KDRestart with Möller-Trumbore");
                    }
                }
                cudaThreadSynchronize();
                cutStopTimer(timerID);
                renderTime = cutGetTimerValue(timerID);
                CHECK_FOR_CUDA_ERROR();
            }

            void RayTracer::HostTrace(int x, int y, TriangleNode* nodes){
                int id = x + y * screenWidth;

                //TriangleNode* nodes = map->GetNodes();
                GeometryList* geom = map->GetGeometry();

                if (map->GetPropagateBoundingBox() == false) leafSkipping = false;

                uchar4 color;
                if (this->intersectionAlgorithm == WOOP){                
                    float4 *woop0, *woop1, *woop2;
                    geom->GetWoopValues(&woop0, &woop1, &woop2);

                    if (leafSkipping){                    
                        color = KDRestart<true, true, true>
                            (id, origin->GetDeviceData(), direction->GetDeviceData(),
                             nodes->GetInfoData(), nodes->GetSplitPositionData(),
                             nodes->GetChildrenData(),
                             nodes->GetPrimitiveIndexData(),
                             nodes->GetPrimitiveBitmapData(),
                             nodes->GetAabbMinData(), nodes->GetAabbMaxData(), 
                             map->GetPrimitiveIndices()->GetDeviceData(),
                             woop0, woop1, woop2,
                             geom->GetNormal0Data(), geom->GetNormal1Data(), geom->GetNormal2Data(),
                             geom->GetColor0Data());
                    }else{
                        color = KDRestart<true, true, false>
                            (id, origin->GetDeviceData(), direction->GetDeviceData(),
                             nodes->GetInfoData(), nodes->GetSplitPositionData(),
                             nodes->GetChildrenData(),
                             nodes->GetPrimitiveIndexData(),
                             nodes->GetPrimitiveBitmapData(),
                             nodes->GetAabbMinData(), nodes->GetAabbMaxData(), 
                             map->GetPrimitiveIndices()->GetDeviceData(),
                             woop0, woop1, woop2,
                             geom->GetNormal0Data(), geom->GetNormal1Data(), geom->GetNormal2Data(),
                             geom->GetColor0Data());
                    }
                }else{
                    if (leafSkipping){
                        color = KDRestart<false, true, true>
                            (id, origin->GetDeviceData(), direction->GetDeviceData(),
                             nodes->GetInfoData(), nodes->GetSplitPositionData(),
                             nodes->GetChildrenData(),
                             nodes->GetPrimitiveIndexData(),
                             nodes->GetPrimitiveBitmapData(),
                             nodes->GetAabbMinData(), nodes->GetAabbMaxData(), 
                             map->GetPrimitiveIndices()->GetDeviceData(),
                             geom->GetP0Data(), geom->GetP1Data(), geom->GetP2Data(),
                             geom->GetNormal0Data(), geom->GetNormal1Data(), geom->GetNormal2Data(),
                             geom->GetColor0Data());
                    }else{
                        color = KDRestart<false, true, false>
                            (id, origin->GetDeviceData(), direction->GetDeviceData(),
                             nodes->GetInfoData(), nodes->GetSplitPositionData(),
                             nodes->GetChildrenData(),
                             nodes->GetPrimitiveIndexData(),
                             nodes->GetPrimitiveBitmapData(),
                             nodes->GetAabbMinData(), nodes->GetAabbMaxData(), 
                             map->GetPrimitiveIndices()->GetDeviceData(),
                             geom->GetP0Data(), geom->GetP1Data(), geom->GetP2Data(),
                             geom->GetNormal0Data(), geom->GetNormal1Data(), geom->GetNormal2Data(),
                             geom->GetColor0Data());
                    }
                }

                logger.info << "Final color: " << make_int4(color.x, color.y, color.z, color.w) << logger.end;
            }

        }
    }
}
