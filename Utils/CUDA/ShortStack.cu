// Short stack raytracer for CUDA
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Utils/CUDA/ShortStack.h>

#include <Display/IViewingVolume.h>
#include <Display/IRenderCanvas.h>
#include <Scene/TriangleNode.h>
#include <Utils/CUDA/Convert.h>
#include <Utils/CUDA/SharedMemory.h>
#include <Utils/CUDA/TriangleMap.h>
#include <Utils/CUDA/Utils.h>
#include <Utils/CUDA/IntersectionTests.h>

#include <Utils/CUDA/LoggerExtensions.h>

#define MAX_THREADS 64
#define MIN_BLOCKS 4
#define SHORT_STACK_SIZE 4

namespace OpenEngine {
    using namespace Display;
    using namespace Resources;
    using namespace Resources::CUDA;
    using namespace Scene;
    namespace Utils {
        namespace CUDA {

#include <Utils/CUDA/Kernels/ColorKernels.h>

            ShortStack::ShortStack(TriangleMap* map)
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

            ShortStack::~ShortStack() {}

            __constant__ int d_rays;

            template <bool invDir>
            __device__ __host__ void TraceNode(float3 origin, float3 direction, 
                                               char axis, float splitPos,
                                               int left, int right, float tMin,
                                               int &node, float &tNext, 
                                               ShortStack::Element* elms, int &nxt, int &cnt){
                            
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
                int lowerChild = 0 < dir ? left : right;
                int upperChild = 0 < dir ? right : left;
                        
                if (tMin < tSplit){
                    node = lowerChild;
                    if (tSplit < tNext)
                        ShortStack::Stack<SHORT_STACK_SIZE>::Push(ShortStack::Element(upperChild, tNext), elms, nxt, cnt);
                    tNext = min(tSplit, tNext);
                }else
                    node = upperChild;

            }

            template <bool useWoop, bool invDir, bool rayBoxIntersect>
            inline __host__ __device__ 
            uchar4 ShortStackTrace(int id, float4* origins, float4* directions,
                                   ShortStack::Element* elms,
                                   char* nodeInfo, float* splitPos,
                                   int2* children,
                                   int *nodePrimIndex, KDNode::bitmap *primBitmap, 
                                   float4* nodeMin, float4* nodeMax,
                                   int *primIndices, 
                                   float4 *v0, float4 *v1, float4 *v2,
                                   float4 *n0s, float4 *n1s, float4 *n2s,
                                   uchar4 *c0s){
                
                int nxt = 0, cnt = 0;
                
                float3 origin = make_float3(FetchGlobalData(origins, id));
                float3 direction = make_float3(FetchGlobalData(directions, id));
                IRayTracer::AdjustRayDirection(direction);

                CUDALogger("=== Ray:  " << origin << " -> " << direction << " ===\n");
                
                float3 tHit;
                tHit.x = 0.0f;
                
                float4 color = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                
                do {
                    int node; float tNext;
                    if (cnt == 0){
                        node = 0;
                        tNext = fInfinity;
                    }else{
                        ShortStack::Element e = ShortStack::Stack<SHORT_STACK_SIZE>::Pop(elms, nxt, cnt);
                        node = e.node;
                        tNext = e.tMax;
                    }
                    char info = FetchGlobalData(nodeInfo, node);
                    
                    if (invDir) direction = make_float3(1.0f, 1.0f, 1.0f) / direction;
                    
                    while ((info & 3) != KDNode::LEAF){
                        CUDALogger("Tracing " << node << " with info " << (int)info);

                        float splitValue = splitPos[node];
                        int2 childPair = children[node];

                        TraceNode<invDir>(origin, direction, info & 3, splitValue, childPair.x, childPair.y, tHit.x,
                                          node, tNext, elms, nxt, cnt);

                        info = nodeInfo[node];
                    }

                    if (invDir) direction = make_float3(1.0f, 1.0f, 1.0f) / direction;
                    tHit.x = tNext * 1.00001f;
                        
                    if (!rayBoxIntersect || 
                        IRayTracer::RayBoxIntersection<true>(origin, make_float3(1.0f) / direction,
                                                             make_float3(FetchGlobalData(nodeMin, node)), 
                                                             make_float3(FetchGlobalData(nodeMax, node)))){
                        
                        int primIndex = FetchGlobalData(nodePrimIndex, node);
                        int primHit = -1;
                        KDNode::bitmap triangles = FetchGlobalData(primBitmap, node);
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
                            float4 newColor = Lighting(tHit, origin, direction, 
                                                       n0s, n1s, n2s,
                                                       c0s, primHit);
                            
                            color = BlendColor(color, newColor);
                            IRayTracer::AdjustRayDirection(direction);
                            
                            // Invalidate the short stack as a new ray has been spawned.
                            ShortStack::Stack<SHORT_STACK_SIZE>::Erase(elms, nxt, cnt);
                            tHit.x = 0.0f;
                        }
                    }
                        
                } while(tHit.x < fInfinity && color.w < 0.97f);

                return make_uchar4(color.x * 255, color.y * 255, color.z * 255, color.w * 255);
            }

            template <bool useWoop, bool invDir, bool rayBoxIntersect>
            __global__ void 
            __launch_bounds__(MAX_THREADS, MIN_BLOCKS) 
                ShortStackKernel(float4* origins, float4* directions,
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

                    ShortStack::Element* elms = SharedMemory<ShortStack::Element>();
                    elms += threadIdx.x * SHORT_STACK_SIZE;
                    //ShortStack::Element elms[SHORT_STACK_SIZE];

                    uchar4 color = ShortStackTrace<useWoop, invDir, rayBoxIntersect>
                        (id, origins, directions, 
                         elms, nodeInfo, splitPos, children,
                         nodePrimIndex, primBitmap, nodeMin, nodeMax, primIndices, 
                         v0, v1, v2, n0s, n1s, n2s, c0s);
                    
                    DumpGlobalData(color, canvas, id);
                }
            }
            
            void ShortStack::Trace(IRenderCanvas* canvas, uchar4* canvasData){
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

                START_TIMER(timerID); 
                KernelConf conf = KernelConf1D(rays, MAX_THREADS, 0, sizeof(Element) * SHORT_STACK_SIZE);
                if (intersectionAlgorithm == WOOP){
                    float4 *woop0, *woop1, *woop2;
                    geom->GetWoopValues(&woop0, &woop1, &woop2);

                    ShortStackKernel<true, true, true><<<conf.blocks, conf.threads, conf.smem>>>
                        (origin->GetDeviceData(), direction->GetDeviceData(),
                         nodes->GetInfoData(), nodes->GetSplitPositionData(),
                         nodes->GetChildrenData(),
                         nodes->GetPrimitiveIndexData(), nodes->GetPrimitiveBitmapData(),
                         nodes->GetAabbMinData(), nodes->GetAabbMaxData(), 
                         map->GetPrimitiveIndices()->GetDeviceData(),
                         woop0, woop1, woop2,
                         geom->GetNormal0Data(), geom->GetNormal1Data(), geom->GetNormal2Data(),
                         geom->GetColor0Data(),
                         canvasData,
                         width);
                    if (printTiming) PRINT_TIMER(timerID, "Short stack using Woop");

                }else{
                    ShortStackKernel<false, true, true><<<conf.blocks, conf.threads, conf.smem>>>
                        (origin->GetDeviceData(), direction->GetDeviceData(),
                         nodes->GetInfoData(), nodes->GetSplitPositionData(),
                         nodes->GetChildrenData(),
                         nodes->GetPrimitiveIndexData(), nodes->GetPrimitiveBitmapData(),
                         nodes->GetAabbMinData(), nodes->GetAabbMaxData(), 
                         map->GetPrimitiveIndices()->GetDeviceData(),
                         geom->GetP0Data(), geom->GetP1Data(), geom->GetP2Data(),
                         geom->GetNormal0Data(), geom->GetNormal1Data(), geom->GetNormal2Data(),
                         geom->GetColor0Data(),
                         canvasData,
                         width);
                    if (printTiming) PRINT_TIMER(timerID, "Short stack using Möller-Trumbore");
                }
                cudaThreadSynchronize();
                cutStopTimer(timerID);
                renderTime = cutGetTimerValue(timerID);
                CHECK_FOR_CUDA_ERROR();                                               
            }
            
            void ShortStack::HostTrace(int x, int y, TriangleNode* nodes){

                int id = x + y * screenWidth;
                
                //TriangleNode* nodes = map->GetNodes();
                GeometryList* geom = map->GetGeometry();

                float4 *woop0, *woop1, *woop2;
                geom->GetWoopValues(&woop0, &woop1, &woop2);

                Element elms[SHORT_STACK_SIZE];

                uchar4 color = ShortStackTrace<true, true, true>
                    (id, origin->GetDeviceData(), direction->GetDeviceData(), elms,
                     nodes->GetInfoData(), nodes->GetSplitPositionData(),
                     nodes->GetChildrenData(),
                     nodes->GetPrimitiveIndexData(),
                     nodes->GetPrimitiveBitmapData(),
                     nodes->GetAabbMinData(), nodes->GetAabbMaxData(), 
                     map->GetPrimitiveIndices()->GetDeviceData(),
                     woop0, woop1, woop2,
                     geom->GetNormal0Data(), geom->GetNormal1Data(), geom->GetNormal2Data(),
                     geom->GetColor0Data());

                logger.info << "Final color: " << make_int4(color.x, color.y, color.z, color.w) << logger.end;

            }

        }
    }
}
