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
