// Raytracer interface for CUDA
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _I_RAY_TRACER_H_
#define _I_RAY_TRACER_H_

#include <Resources/CUDA/CUDADataBlock.h>

namespace OpenEngine {
    namespace Display {
        class IRenderCanvas;
    }
    namespace Scene {
        class TriangleNode;
    }
    namespace Utils {
        namespace CUDA {
            
            class IRayTracer {
            public:
                enum IntersectionAlgorithm {MOELLER, WOOP};

            protected:
                bool visualizeRays;
                IntersectionAlgorithm intersectionAlgorithm;
                
                Resources::CUDA::CUDADataBlock<1, float4> *origin;
                Resources::CUDA::CUDADataBlock<1, float4> *direction;
                
            public:
                IRayTracer();
                virtual ~IRayTracer();

                virtual void Trace(Display::IRenderCanvas* canvas, uchar4* canvasData) = 0;
                virtual void HostTrace(float3 origin, float3 direction, Scene::TriangleNode* nodes) = 0;

                void SetVisualizeRays(const bool v) {visualizeRays = v;}
                void SetIntersectionAlgorithm(const IntersectionAlgorithm a) { intersectionAlgorithm = a;}

#define PW 4
#define PH 8
                static inline __device__ __host__ int PacketIndex(int id, int screenWidth){
                    int cell = id / (PW * PH);
                    int bx = cell % (screenWidth / PW);
                    int by = cell / (screenWidth / PW);
                    int x = id % PW;
                    int y = (id % (PW * PH)) / PW;
                    return bx * PW + x + (by * PH + y) * screenWidth;
                }
                
            protected:
                void CreateInitialRays(Display::IRenderCanvas* canvas);
                void RenderRays(uchar4 *canvas, int rays);
            };

        }
    }
}

#endif
