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
                    float tMax;

                    __device__ __host__ Element()
                        : node(0), tMax(0.0f) {}
                    __device__ __host__ Element(const int node, const float tMax)
                        : node(node), tMax(tMax) {}

                    __host__ std::string ToString() const {
                        std::ostringstream out;
                        out << "{node: " << node << ", tMax: " << tMax << "}";
                        return out.str();
                    }
                };

                template <int N> struct Stack {
                    Element elm[N];
                    int next, count;

                    __device__ __host__ Stack() 
                        : next(0), count(0) {}

                    __device__ __host__ void Erase() { next = count = 0; }

                    __device__ __host__ bool IsEmpty() const { return count == 0; }
                    
                    __device__ __host__ void Push(const Element e) {
                        elm[next] = e;
                        next = next == N-1 ? 0 : next+1;
                        count = count == N ? N : count+1;
                    }
                    
                    __device__ __host__ Element Pop() { 
                        next = (next == 0 ? N : next) - 1;
                        count--;
                        return elm[next];
                    }

                    __host__ std::string ToString() const{
                        std::ostringstream out;
                        int e = next -1;
                        if (e == -1) e = N-1;
                        int cnt = count;
                        out << "Stack: [";
                        if (0 < cnt)
                            out << elm[e].ToString();
                        for (int i = 1; i < cnt; ++i){
                            e--; if (e == -1) e = N-1;
                            out << ", " << elm[e].ToString();
                        }
                        out << "]\n";
                        //out << "count: " << count << ", next: " << next;
                        return out.str();
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
