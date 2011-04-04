// CUDA Mesh Node.
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Scene/CUDAMeshNode.h>

#include <Geometry/GeometrySet.h>
#include <Geometry/Mesh.h>
#include <Resources/IDataBlock.h>
#include <Scene/MeshNode.h>
#include <Utils/CUDA/Convert.h>
#include <Utils/CUDA/Utils.h>

namespace OpenEngine {
    using namespace Resources;
    using namespace Resources::CUDA;
    namespace Scene {

        __global__ void CopyVertices(float3* vertIn,
                                     float4* vertOut,
                                     const int size){

            const int id = blockDim.x * blockIdx.x + threadIdx.x;

            if (id < size){
                vertOut[id] = make_float4(vertIn[id], 1.0);
            }
        }

        __global__ void CopyNormals(float3* normIn,
                                    float4* normOut,
                                    const int size){

            const int id = blockDim.x * blockIdx.x + threadIdx.x;

            if (id < size){
                normOut[id] = make_float4(normIn[id], 0.0);
            }
        }

        __global__ void CopyColors(float3* colorIn, uchar4 *colorOut, const int size){
            const int id = blockDim.x * blockIdx.x + threadIdx.x;
                
            if (id < size){
                float3 c = colorIn[id];
                colorOut[id] = make_uchar4(c.x * 255, c.y * 255, c.z * 255, 255);
            }
        }

        __global__ void CopyColors(float4* colorIn, uchar4 *colorOut, const int size){
            const int id = blockDim.x * blockIdx.x + threadIdx.x;
                
            if (id < size){
                float4 c = colorIn[id];
                colorOut[id] = make_uchar4(c.x * 255, c.y * 255, c.z * 255, c.w * 255);
            }
        }

        __global__ void SetColor(const uchar4 color, uchar4* colorOut, const int size){
            const int id = blockDim.x * blockIdx.x + threadIdx.x;
            
            if (id < size){
                colorOut[id] = color;
            }
        }

        CUDAMeshNode::CUDAMeshNode(MeshNode* mesh){
            IDataBlockPtr v = mesh->GetMesh()->GetGeometrySet()->GetVertices();
            IDataBlockPtr n = mesh->GetMesh()->GetGeometrySet()->GetNormals();
            IDataBlockPtr c = mesh->GetMesh()->GetGeometrySet()->GetColors();
            IndicesPtr i = mesh->GetMesh()->GetIndices();

            unsigned int size = v->GetSize();
                
            unsigned int blocks, threads;
            Calc1DKernelDimensions(size, blocks, threads, 128);

            // @TODO change to use mapped memory

            float3 *hat;
            cudaMalloc(&hat, size * sizeof(float3));

            vertices = new CUDADataBlock<float4>(size);
            if (v->GetDimension() == 3){
                cudaMemcpy(hat, v->GetVoidDataPtr(), size * sizeof(float3), cudaMemcpyHostToDevice);
                CHECK_FOR_CUDA_ERROR();
                CopyVertices<<<blocks, threads>>>(hat, vertices->GetDeviceData(), size);
                CHECK_FOR_CUDA_ERROR();
            }else if (v->GetDimension() == 4){
                cudaMemcpy(vertices->GetDeviceData(), v->GetVoidData(), size * sizeof(float4), cudaMemcpyHostToDevice);
                CHECK_FOR_CUDA_ERROR();
            }else
                throw Exception("Deux the fuck");

            normals = new CUDADataBlock<float4>(size);
            if (n->GetDimension() == 3){
                cudaMemcpy(hat, n->GetVoidDataPtr(), size * sizeof(float3), cudaMemcpyHostToDevice);
                CHECK_FOR_CUDA_ERROR();
                CopyNormals<<<blocks, threads>>>(hat, normals->GetDeviceData(), size);
                CHECK_FOR_CUDA_ERROR();
            }else
                throw Exception("Quad the fuck");
            cudaFree(hat);

            colors = new CUDADataBlock<uchar4>(size);
            if (c != NULL){
                if (c->GetDimension() == 3){
                    float3 *hat;
                    cudaMalloc(&hat, size * sizeof(float3));
                    
                    cudaMemcpy(hat, c->GetVoidDataPtr(), size * sizeof(float3), cudaMemcpyHostToDevice);
                    CHECK_FOR_CUDA_ERROR();
                    CopyColors<<<blocks, threads>>>(hat, colors->GetDeviceData(), size);
                    CHECK_FOR_CUDA_ERROR();
                    
                    cudaFree(hat);
                }else if (c->GetDimension() == 4){
                    float4 *hat;
                    cudaMalloc(&hat, size * sizeof(float4));                    
                    
                    cudaMemcpy(hat, c->GetVoidDataPtr(), size * sizeof(float4), cudaMemcpyHostToDevice);
                    CHECK_FOR_CUDA_ERROR();
                    CopyColors<<<blocks, threads>>>(hat, colors->GetDeviceData(), size);
                    CHECK_FOR_CUDA_ERROR();
                    cudaFree(hat);
                }
                CHECK_FOR_CUDA_ERROR();
            }else
                SetColor<<<blocks, threads>>>(make_uchar4(180, 180, 180, 255), colors->GetDeviceData(), size);
            
            indices = new CUDADataBlock<unsigned int>(i->GetSize());
            cudaMemcpy(indices->GetDeviceData(), i->GetData(), i->GetSize() * sizeof(unsigned int), cudaMemcpyHostToDevice);
            CHECK_FOR_CUDA_ERROR();
        }
            
    }
}
