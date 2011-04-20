// Oscillating CUDA Mesh Node.
// -------------------------------------------------------------------
// Copyright (C) 2011 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Scene/OscCUDAMeshNode.h>
#include <Meta/CUDA.h>
#include <Utils/CUDA/LoggerExtensions.h>

using namespace OpenEngine::Resources::CUDA;

namespace OpenEngine {
    namespace Scene {

        OscCUDAMeshNode::OscCUDAMeshNode(float width, float depth, float height, int detail) 
            : width(width), depth(depth), height(height), detail(detail), time(0) {
            this->vertices = NULL;
            this->normals = NULL;
            this->colors = NULL;
            this->indices = NULL;
        }
        
        void OscCUDAMeshNode::Init(){
            if (vertices) return;
            
            unsigned int d = detail + 1;
            unsigned int size = d * d;
            float2 unitSize = make_float2(width / float(detail), depth / float(detail));
                
            float4 vertex[size];
            for (int i = 0; i < d; ++i)
                for (int j = 0; j < d; ++j){
                    int index = i + j * d;
                    vertex[index] = make_float4(i * unitSize.x - width * 0.5f, height,
                                                j * unitSize.y - depth * 0.5f, 1.0f);
                }
            this->vertices = new CUDADataBlock<float4>(size, vertex);
            
            // Normals will be set at render time.
            float4 hat[size];
            for (int i = 0; i < size; ++i)
                hat[i] = make_float4(0.0f, 1.0f, .0f, .0f);
            this->normals = new CUDADataBlock<float4>(size, hat);

            uchar4 color[size];
            for (int i = 0; i < size; ++i)
                color[i] = make_uchar4(100, 200, 230, 128);
            this->colors = new CUDADataBlock<uchar4>(size, color);

            unsigned int i[6 * detail * detail];
            unsigned int index = 0;
            for (unsigned int m = 0; m < detail; ++m){
                for (unsigned int n = 0; n < detail; ++n){
                    // Index the (i, j)'th quad of 2 triangles
                    i[index++] = m + n * d;
                    i[index++] = m + (n+1) * d;
                    i[index++] = m+1 + n * d;
                    
                    i[index++] = m+1 + n * d;
                    i[index++] = m + (n+1) * d;
                    i[index++] = m+1 + (n+1) * d;
                }
            }

            this->indices = new CUDADataBlock<unsigned int>(6*detail*detail, i);
        }

        const float amplitude = 0.2f;
        const float frequenz = 0.3f;

        __global__
        void UpdateSurface(float4* vert, float4* norm, 
                           int detail, float height, float time){
            int x = blockDim.x * blockIdx.x + threadIdx.x;
            int y = blockDim.y * blockIdx.y + threadIdx.y;
            if (x < detail && y < detail){
                int id = y * detail + x;
                float4 v = vert[id];
                v.y = height + amplitude * cos(frequenz * float(x) + time);
                v.y += amplitude * sin(frequenz * float(y) + time);
                vert[id] = v;

                float4 n;
                n.x = -amplitude * sin(frequenz * float(x) + time) * frequenz;
                n.z = amplitude * cos(frequenz * float(y) + time) * frequenz;
                n.y = 1.0f - n.x*n.x - n.z*n.z;
                n.w = 0.0f;
                norm[id] = n;
            }
        }

        void OscCUDAMeshNode::Handle(Core::ProcessEventArg arg){
            time += arg.approx;

            dim3 blockSize(16, 16, 1);
            dim3 gridSize(ceil(float(detail)/float(blockSize.x)), 
                          ceil(float(detail)/float(blockSize.y)), 1);

            UpdateSurface<<<gridSize, blockSize>>>(this->vertices->GetDeviceData(), 
                                                   this->normals->GetDeviceData(),
                                                   detail+1, height, time/700000.0f);
            CHECK_FOR_CUDA_ERROR();
        }
    }
}
