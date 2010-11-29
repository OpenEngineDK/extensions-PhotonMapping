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

#define MAX_THREADS 128
#define SHORT_STACK_SIZE 1

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

            __global__ void
            __launch_bounds__(MAX_THREADS) 
                Hat(int* hut){
                const int id = blockDim.x * blockIdx.x + threadIdx.x;                
                
                ShortStack::Element* elms = SharedMemory<ShortStack::Element>();
                int offset = id * SHORT_STACK_SIZE;
                int nxt = 0, cnt = 0;
                
                ShortStack::Stack<SHORT_STACK_SIZE>::Push(ShortStack::Element(id, 10.0f), elms + offset, nxt, cnt);

                ShortStack::Element e = ShortStack::Stack<SHORT_STACK_SIZE>::Pop(elms + offset, nxt, cnt);
                
                hut[id] = e.node;
            }
            
            __global__ void 
            __launch_bounds__(MAX_THREADS) 
                ShortStackTrace(float4* origins, float4* directions,
                                char* nodeInfo, float* splitPos,
                                int2* children,
                                int *nodePrimIndex, KDNode::bitmap *primBitmap, 
                                int *primIndices, 
                                float4 *v0, float4 *v1, float4 *v2,
                                float4 *n0s, float4 *n1s, float4 *n2s,
                                uchar4 *c0s,
                                uchar4 *canvas){
                
                const int id = blockDim.x * blockIdx.x + threadIdx.x;
                
                if (id < d_rays){
                    
                    //ShortStack::Stack<6> *stack = SharedMemory<ShortStack::Stack<6> >();
                    //stack += threadIdx.x;
                    //ShortStack::Element *elms = SharedMemory<ShortStack::Element>();
                    //ShortStack::Stack<SHORT_STACK_SIZE> stack(elms + threadIdx.x * SHORT_STACK_SIZE);
                    //ShortStack::Stack<SHORT_STACK_SIZE> stack;
                    
                    //ShortStack::Element* elms = SharedMemory<ShortStack::Element>();
                    //elms += id * SHORT_STACK_SIZE;
                    ShortStack::Element elms[SHORT_STACK_SIZE];
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
                        
                            float tSplit = (splitValue - ori) / dir;
                            int lowerChild = 0 < dir ? childPair.x : childPair.y;
                            int upperChild = 0 < dir ? childPair.y : childPair.x;
                        
                            if (tHit.x < tSplit){
                                node = lowerChild;
                                if (tSplit < tNext)
                                    ShortStack::Stack<SHORT_STACK_SIZE>::Push(ShortStack::Element(upperChild, tNext), elms, nxt, cnt);
                                tNext = min(tSplit, tNext);
                            }else
                                node = upperChild;

                            info = nodeInfo[node];
                        }
                        
                        tHit.x = tNext;
                        
                        int primIndex = nodePrimIndex[node];
                        int primHit = -1;
                        KDNode::bitmap triangles = primBitmap[node];
                        while (triangles){
                            int i = __ffs(triangles) - 1;
                            
                            int prim = primIndices[primIndex + i];

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

                            // Invalidate the short stack as a new ray has been spawned.
                            ShortStack::Stack<SHORT_STACK_SIZE>::Erase(nxt, cnt);
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

                unsigned int blocks, threads, smemSize;
                unsigned int smemPrThread = sizeof(Element) * SHORT_STACK_SIZE;
                Calc1DKernelDimensionsWithSmem(rays, smemPrThread, blocks, threads, smemSize, MAX_THREADS);
                START_TIMER(timerID); 
                /*
                Hat<<<1, threads, smemSize>>>(map->GetPrimitiveIndices()->GetDeviceData());
                
                logger.info << Convert::ToString(map->GetPrimitiveIndices()->GetDeviceData(), threads) << logger.end;
                exit(0);
                */
                logger.info << "ShortStackTrace<<<" << blocks << ", " << threads << ", " << smemSize << ">>>" << logger.end;
                ShortStackTrace<<<blocks, threads, smemSize>>>(origin->GetDeviceData(), direction->GetDeviceData(),
                                                               nodes->GetInfoData(), nodes->GetSplitPositionData(),
                                                               nodes->GetChildrenData(),
                                                               nodes->GetPrimitiveIndexData(), nodes->GetPrimitiveBitmapData(),
                                                               map->GetPrimitiveIndices()->GetDeviceData(),
                                                               geom->GetP0Data(), geom->GetP1Data(), geom->GetP2Data(),
                                                               geom->GetNormal0Data(), geom->GetNormal1Data(), geom->GetNormal2Data(),
                                                               geom->GetColor0Data(),
                                                               canvasData);
                PRINT_TIMER(timerID, "Short stack");
                CHECK_FOR_CUDA_ERROR();                                               
            }
            
            void ShortStack::HostTrace(float3 origin, float3 direction, TriangleNode* nodes){
                GeometryList* geom = map->GetGeometry();

                //Stack<SHORT_STACK_SIZE> stack;
                Element elms[SHORT_STACK_SIZE];
                int next = 0, count = 0;
                
                float3 tHit;
                tHit.x = 0.0f;

                float4 color = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

                do {
                    logger.info << "=== Ray:  " << Convert::ToString(origin) << " -> " << Convert::ToString(direction) << " ===" << logger.end;
                    logger.info << Stack<SHORT_STACK_SIZE>::ToString(elms, next, count) << logger.end;
                    
                    int node; float tNext;
                    //if (stack.IsEmpty()){
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
                        
                        float tSplit = (splitValue - ori) / dir;
                        int lowerChild = 0 < dir ? childPair.x : childPair.y;
                        int upperChild = 0 < dir ? childPair.y : childPair.x;

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
                        
                        triangles -= 1<<i;
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

                        // Invalidate the shortstack as we're now tracing a new ray.
                        Stack<SHORT_STACK_SIZE>::Erase(next, count);
                        tHit.x = 0.0f;
                    }

                } while(tHit.x < fInfinity && color.w < 0.97f);

            }

        }
    }
}
