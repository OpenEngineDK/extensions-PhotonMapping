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
#include <Utils/CUDA/TriangleMap.h>
#include <Display/IViewingVolume.h>
#include <Display/IRenderCanvas.h>
#include <Resources/CUDA/CUDADataBlock.h>

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {
            
            class RayTracer : public IRayTracer {
            public:
                TriangleMap *map;
                Resources::CUDA::CUDADataBlock<1, float4> *origin;
                Resources::CUDA::CUDADataBlock<1, float4> *dir;
                
            public:
                RayTracer(TriangleMap* map);
                virtual ~RayTracer() {}

                void Trace(Display::IRenderCanvas* canvas, uchar4* canvasData);
            };

        }
    }
}

#endif
