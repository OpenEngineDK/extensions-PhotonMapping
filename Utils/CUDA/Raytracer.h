// Raytracer class for CUDA
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _CUDA_RAY_TRACER_H_
#define _CUDA_RAY_TRACER_H_

#include <Utils/CUDA/IRayTracer.h>

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {

            class TriangleMap;
            
            class RayTracer : public IRayTracer {
            public:
                unsigned int timerID;

                TriangleMap *map;
                
            public:
                RayTracer(TriangleMap* map);
                virtual ~RayTracer();

                void Trace(Display::IRenderCanvas* canvas, uchar4* canvasData);

                void HostTrace(float3 origin, float3 direction, Scene::TriangleNode* nodes);
            };

        }
    }
}

#endif
