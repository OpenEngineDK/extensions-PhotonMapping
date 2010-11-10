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

namespace OpenEngine {
    namespace Display {
        class IRenderCanvas;
    }
    namespace Utils {
        namespace CUDA {
            
            class IRayTracer {
            protected:
                bool visualizeRays;
                
            public:
                IRayTracer() : visualizeRays(false) {}
                virtual ~IRayTracer() {}

                virtual void Trace(Display::IRenderCanvas* canvas, uchar4* canvasData) = 0;

                void SetVisualizeRays(const bool v) {visualizeRays = v;}
            };

        }
    }
}

#endif
