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

#include <sstream>

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {

            class TriangleMap;

            class ShortStack : public IRayTracer {
            public:
                struct Element {
                    int node;
                    float tMin, tMax;

                    __device__ __host__ Element()
                        : node(0), tMin(0.0f), tMax(0.0f) {}
                    __device__ __host__ Element(int node, float tMin, float tMax)
                        : node(node), tMin(tMin), tMax(tMax) {}

                    __host__ std::string ToString() {
                        std::ostringstream out;
                        out << "{node: " << node << ", tMin: " << tMin << ", tMax: " << tMax << "}";
                        return out.str();
                    }
                };

                template <int N> struct Stack {
                    Element elm[N];
                    int next;
                    int count;

                    __device__ __host__ Stack() 
                        : next(0), count(0) {}

                    __device__ __host__ bool Empty() { return count == 0; }
                    
                    // @OPT replace if with modulo or ? :

                    __device__ __host__ void Push(Element e) {
                        elm[next] = e;
                        next++;
                        if (next == N) next = 0;
                        count++;
                        if (count > N) count = N;
                    }
                    
                    __device__ __host__ Element Pop() { 
                        next--;
                        if (next == -1) next = N-1;
                        count--;
                        return elm[next];
                    }
                };
                
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
