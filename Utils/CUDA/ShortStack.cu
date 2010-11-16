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
#include <Utils/CUDA/TriangleMap.h>
#include <Utils/CUDA/Utils.h>

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

            __global__ void ShortStackTrace(float4* origins, float4* directions,
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

                unsigned int blocks, threads;
                Calc1DKernelDimensions(rays, blocks, threads, 128);
                START_TIMER(timerID); 
                ShortStackTrace<<<blocks, threads>>>(origin->GetDeviceData(), direction->GetDeviceData(),
                                                     nodes->GetInfoData(), nodes->GetSplitPositionData(),
                                                     nodes->GetLeftData(), nodes->GetRightData(),
                                                     nodes->GetPrimitiveInfoData(),
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

                Stack<8> stack;

                logger.info << "Origin " << Convert::ToString(origin) << logger.end;
                logger.info << "Direction " << Convert::ToString(direction) << "\n" << logger.end;

                float3 tHit;
                tHit.x = 0.0f;
            }

        }
    }
}
