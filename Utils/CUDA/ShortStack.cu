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

#define MAX_THREADS 64
#define MIN_BLOCKS 3
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
                        
                float tSplit = (splitPos - ori) / dir;
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

            template <bool useWoop>
            __global__ void 
            __launch_bounds__(MAX_THREADS, MIN_BLOCKS) 
                ShortStackTrace(float4* origins, float4* directions,
                                char* nodeInfo, float* splitPos,
                                int2* children,
                                int *nodePrimIndex, KDNode::bitmap *primBitmap, 
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
                    int nxt = 0, cnt = 0;

                    float3 origin = make_float3(origins[id]);
                    float3 direction = make_float3(directions[id]);

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
                        
                        char info = nodeInfo[node];

                        while ((info & 3) != KDNode::LEAF){
                            
                            float splitValue = splitPos[node];
                            int2 childPair = children[node];

                            TraceNode(origin, direction, info & 3, splitValue, childPair.x, childPair.y, tHit.x,
                                      node, tNext, elms, nxt, cnt);

                            info = nodeInfo[node];
                        }
                 
                        tHit.x = tNext;
                        
                        int primIndex = nodePrimIndex[node];
                        int primHit = -1;
                        KDNode::bitmap triangles = primBitmap[node];
                        while (triangles){
                            int i = __ffs(triangles) - 1;
                            
                            int prim = primIndices[primIndex + i];

                            if (useWoop){
                                IRayTracer::Woop(v0, v1, v2, prim,
                                                 origin, direction, primHit, tHit);
                            }else{
                                IRayTracer::MoellerTrumbore(v0, v1, v2, prim,
                                                            origin, direction, primHit, tHit);
                            }
                            
                            triangles -= 1<<i;
                        }

                        if (primHit != -1){
                            float4 newColor = Lighting(tHit, origin, direction, 
                                                       n0s[primHit], n1s[primHit], n2s[primHit],
                                                       c0s[primHit]);
                            
                            color = BlendColor(color, newColor);

                            // Invalidate the short stack as a new ray has been spawned.
                            ShortStack::Stack<SHORT_STACK_SIZE>::Erase(elms, nxt, cnt);
                            tHit.x = 0.0f;
                        }
                        
                    } while(tHit.x < fInfinity && color.w < 0.97f);

                    canvas[id] = make_uchar4(color.x * 255, color.y * 255, color.z * 255, color.w * 255);
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

                if (intersectionAlgorithm == WOOP){
                    float4 *woop0, *woop1, *woop2;
                    geom->GetWoopValues(&woop0, &woop1, &woop2);

                    KernelConf conf = KernelConf1D(rays, MAX_THREADS, 0, sizeof(Element) * SHORT_STACK_SIZE);
                    START_TIMER(timerID); 
                    ShortStackTrace<true><<<conf.blocks, conf.threads, conf.smem>>>
                        (origin->GetDeviceData(), direction->GetDeviceData(),
                         nodes->GetInfoData(), nodes->GetSplitPositionData(),
                         nodes->GetChildrenData(),
                         nodes->GetPrimitiveIndexData(), nodes->GetPrimitiveBitmapData(),
                         map->GetPrimitiveIndices()->GetDeviceData(),
                         woop0, woop1, woop2,
                         geom->GetNormal0Data(), geom->GetNormal1Data(), geom->GetNormal2Data(),
                         geom->GetColor0Data(),
                         canvasData,
                         width);
                    PRINT_TIMER(timerID, "Short stack using Woop");

                }else{
                    unsigned int blocks, threads, smemSize;
                    unsigned int smemPrThread = sizeof(Element) * SHORT_STACK_SIZE;
                    Calc1DKernelDimensionsWithSmem(rays, smemPrThread, blocks, threads, smemSize, MAX_THREADS);
                    START_TIMER(timerID); 
                    ShortStackTrace<false><<<blocks, threads, smemSize>>>
                        (origin->GetDeviceData(), direction->GetDeviceData(),
                         nodes->GetInfoData(), nodes->GetSplitPositionData(),
                         nodes->GetChildrenData(),
                         nodes->GetPrimitiveIndexData(), nodes->GetPrimitiveBitmapData(),
                         map->GetPrimitiveIndices()->GetDeviceData(),
                         geom->GetP0Data(), geom->GetP1Data(), geom->GetP2Data(),
                         geom->GetNormal0Data(), geom->GetNormal1Data(), geom->GetNormal2Data(),
                         geom->GetColor0Data(),
                         canvasData,
                         width);
                    PRINT_TIMER(timerID, "Short stack using Möller-Trumbore");
                }
                CHECK_FOR_CUDA_ERROR();                                               
            }
            
            void ShortStack::HostTrace(int x, int y, TriangleNode* nodes){

                int id = x + y * screenWidth;
                float3 ori, dir;
                cudaMemcpy(&ori, origin->GetDeviceData() + id, sizeof(float3), cudaMemcpyDeviceToHost);
                cudaMemcpy(&dir, direction->GetDeviceData() + id, sizeof(float3), cudaMemcpyDeviceToHost);

                GeometryList* geom = map->GetGeometry();

                Element elms[SHORT_STACK_SIZE];
                int next = 0, count = 0;
                
                float3 tHit;
                tHit.x = 0.0f;

                float4 color = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

                do {
                    logger.info << "=== Ray:  " << Utils::CUDA::Convert::ToString(ori) << " -> " << Convert::ToString(dir) << " ===" << logger.end;
                    logger.info << Stack<SHORT_STACK_SIZE>::ToString(elms, next, count) << logger.end;
                    
                    int node; float tNext;
                    if (count == 0){
                        node = 0;
                        tNext = fInfinity;
                    }else{
                        Element e = Stack<SHORT_STACK_SIZE>::Pop(elms, next, count);
                        node = e.node;
                        tNext = e.tMax;
                    }

                    char info;
                    cudaMemcpy(&info, nodes->GetInfoData() + node, sizeof(char), cudaMemcpyDeviceToHost);
                    CHECK_FOR_CUDA_ERROR();

                    while ((info & 3) != KDNode::LEAF){
                        logger.info << "Tracing " << node << " with info " << (int)info << logger.end;
                        
                        float splitValue;
                        cudaMemcpy(&splitValue, nodes->GetSplitPositionData() + node, sizeof(float), cudaMemcpyDeviceToHost);
                        CHECK_FOR_CUDA_ERROR();
                        
                        int2 childPair;
                        cudaMemcpy(&childPair, nodes->GetChildrenData() + node, sizeof(int2), cudaMemcpyDeviceToHost);
                        CHECK_FOR_CUDA_ERROR();

                        // Trace
                        float origin, direction;
                        switch(info & 3){
                        case KDNode::X:
                            origin = ori.x; direction = dir.x;
                            break;
                        case KDNode::Y:
                            origin = ori.y; direction = dir.y;
                            break;
                        case KDNode::Z:
                            origin = ori.z; direction = dir.z;
                            break;
                        }
                        
                        float tSplit = (splitValue - origin) / direction;
                        int lowerChild = 0 < direction ? childPair.x : childPair.y;
                        int upperChild = 0 < direction ? childPair.y : childPair.x;

                        if (tHit.x < tSplit){
                            node = lowerChild;
                            if (tSplit < tNext)
                                Stack<SHORT_STACK_SIZE>::Push(Element(upperChild, tNext), elms, next, count);
                            tNext = min(tSplit, tNext);
                        }else
                            node = upperChild;
                        
                        // New nodes info
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

                        if (intersectionAlgorithm == WOOP){
                            
                            float4 *woop0, *woop1, *woop2;
                            geom->GetWoopValues(&woop0, &woop1, &woop2);

                            IRayTracer::Woop(woop0, woop1, woop2, prim,
                                             ori, dir, primHit, tHit);

                        }else{

                            IRayTracer::MoellerTrumbore(geom->GetP0Data(), geom->GetP1Data(), geom->GetP2Data(), prim,
                                                        ori, dir, primHit, tHit);
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
                        
                        float4 newColor = Lighting(tHit, ori, dir, n0, n1, n2, c0);

                        logger.info << "New color: " << Convert::ToString(newColor) << logger.end;
                        
                        color = BlendColor(color, newColor);

                        logger.info << "Color: " << Convert::ToString(color) << "\n" << logger.end;

                        // Invalidate the shortstack as we're now tracing a new ray.
                        Stack<SHORT_STACK_SIZE>::Erase(elms, next, count);
                        tHit.x = 0.0f;
                    }

                } while(tHit.x < fInfinity && color.w < 0.97f);

            }

        }
    }
}
