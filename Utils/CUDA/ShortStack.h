// Short stack raytracer for CUDA
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _CUDA_SHORT_STACK_RAY_TRACER_H_
#define _CUDA_SHORT_STACK_RAY_TRACER_H_

#include <Utils/CUDA/IRayTracer.h>

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {

            class TriangleMap;

            class ShortStack : public IRayTracer {
            protected:
                unsigned int timerID;

                TriangleMap *map;

            public:
                ShortStack(TriangleMap *map);
                virtual ~ShortStack();

                void Trace(Display::IRenderCanvas* canvas, uchar4* canvasData);
                void HostTrace(float3 origin, float3 direction, Scene::TriangleNode* nodes);

            };
            
        }
    }
}

#endif
