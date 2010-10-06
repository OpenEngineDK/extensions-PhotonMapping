// Geometry List.
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Utils/CUDA/GeometryList.h>

#include <Geometry/Mesh.h>
#include <Geometry/GeometrySet.h>
#include <Scene/ISceneNode.h>
#include <Scene/MeshNode.h>
#include <Scene/TransformationNode.h>
#include <Utils/CUDA/Utils.h>
#include <Math/CUDA/Matrix.h>

using namespace OpenEngine::Geometry;
using namespace OpenEngine::Math;
using namespace OpenEngine::Math::CUDA;
using namespace OpenEngine::Scene;
using namespace OpenEngine::Resources::CUDA;

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {

            GeometryList::GeometryList()
                : maxSize(0) {}

            GeometryList::GeometryList(int size)
                : maxSize(size) {

                p0 = new CUDADataBlock<1, float4>(maxSize);
                p1 = new CUDADataBlock<1, float4>(maxSize);
                p2 = new CUDADataBlock<1, float4>(maxSize);

                n0 = new CUDADataBlock<1, float4>(maxSize);
                n1 = new CUDADataBlock<1, float4>(maxSize);
                n2 = new CUDADataBlock<1, float4>(maxSize);

                c0 = new CUDADataBlock<1, uchar4>(maxSize);
                c1 = new CUDADataBlock<1, uchar4>(maxSize);
                c2 = new CUDADataBlock<1, uchar4>(maxSize);

                aabbMin = new CUDADataBlock<1, float4>(maxSize);
                aabbMax = new CUDADataBlock<1, float4>(maxSize);

                surfaceArea = new CUDADataBlock<1, float>(maxSize);
            }

            void GeometryList::Resize(int size){
                
            }

            __global__ void AddMesh(unsigned int *indices,
                                    float3 *verticesIn,
                                    float3 *normalsIn,
                                    float4 *colorsIn,
                                    Matrix44f modelMat,
                                    Matrix33f normalMat,
                                    float4 *p0, float4 *p1, float4 *p2,
                                    float4 *n0, float4 *n1, float4 *n2,
                                    uchar4 *c0, uchar4 *c1, uchar4 *c2,
                                    float *surfaceArea,
                                    int size){
                
                const int id = blockDim.x * blockIdx.x + threadIdx.x;

                if (id < size){
                    const int i = __mul24(id, 3);
                    const unsigned int i0 = indices[i];
                    const unsigned int i1 = indices[i+1];
                    const unsigned int i2 = indices[i+2];
                    
                    const float3 v0 = verticesIn[i0];
                    const float3 v1 = verticesIn[i1];
                    const float3 v2 = verticesIn[i2];
                    surfaceArea[id] = 0.5f * length(cross(v1-v0, v2-v0));
                    
                    p0[id] = modelMat * make_float4(v0, 1.0f);
                    p1[id] = modelMat * make_float4(v1, 1.0f);
                    p2[id] = modelMat * make_float4(v2, 1.0f);
                    
                    n0[id] = make_float4(normalMat * normalsIn[i0], 0);
                    n1[id] = make_float4(normalMat * normalsIn[i1], 0);
                    n2[id] = make_float4(normalMat * normalsIn[i2], 0);

                    c0[id] = make_uchar4(colorsIn[i0].x * 255.0f, colorsIn[i0].y * 255.0f, colorsIn[i0].z * 255.0f, colorsIn[i0].w * 255.0f);
                    c1[id] = make_uchar4(colorsIn[i1].x * 255.0f, colorsIn[i1].y * 255.0f, colorsIn[i1].z * 255.0f, colorsIn[i1].w * 255.0f);
                    c2[id] = make_uchar4(colorsIn[i2].x * 255.0f, colorsIn[i2].y * 255.0f, colorsIn[i2].z * 255.0f, colorsIn[i2].w * 255.0f);
                }
            }

            void GeometryList::AddMesh(MeshPtr mesh, Matrix<4,4,float> modelMat){
                GeometrySetPtr geom = mesh->GetGeometrySet();
                if (geom->GetDataBlock("vertex") && geom->GetDataBlock("vertex")->GetID() != 0){
                    // Geometry has been loaded to the graphics card
                    // and we can copy it from there.
                    IndicesPtr indices = mesh->GetIndices();
                    IDataBlockPtr vertices = geom->GetDataBlock("vertex");
                    IDataBlockPtr normals = geom->GetDataBlock("normal");
                    IDataBlockPtr colors = geom->GetDataBlock("color");

                    if (p0->GetSize() + indices->GetSize() < maxSize)
                        Resize(p0->GetSize() + indices->GetSize());
                    
                    cudaGraphicsResource *iResource, *vResource, *nResource, *cResource;
                    cudaGraphicsGLRegisterBuffer(&iResource, indices->GetID(), cudaGraphicsMapFlagsReadOnly);
                    cudaGraphicsGLRegisterBuffer(&vResource, vertices->GetID(), cudaGraphicsMapFlagsReadOnly);
                    cudaGraphicsGLRegisterBuffer(&nResource, normals->GetID(), cudaGraphicsMapFlagsReadOnly);
                    cudaGraphicsGLRegisterBuffer(&cResource, colors->GetID(), cudaGraphicsMapFlagsReadOnly);
                    CHECK_FOR_CUDA_ERROR();
                    
                    size_t bytes;
                    unsigned int* in;
                    cudaGraphicsResourceGetMappedPointer((void**)&in, &bytes,
                                                         iResource);
                    float3* pos;
                    cudaGraphicsResourceGetMappedPointer((void**)&pos, &bytes,
                                                         vResource);
                    float3* norms;
                    cudaGraphicsResourceGetMappedPointer((void**)&norms, &bytes,
                                                         nResource);
                    float4* cols;
                    cudaGraphicsResourceGetMappedPointer((void**)&cols, &bytes,
                                                         cResource);

                    unsigned int blocks, threads;
                    Calc1DKernelDimensions(indices->GetSize(), blocks, threads);
                    Math::CUDA::Matrix44f mat = Math::CUDA::Matrix44f(modelMat);
                    //AddMesh();

                    cudaGraphicsUnmapResources(1, &iResource, 0);
                    cudaGraphicsUnmapResources(1, &vResource, 0);
                    cudaGraphicsUnmapResources(1, &nResource, 0);
                    cudaGraphicsUnmapResources(1, &cResource, 0);
                    CHECK_FOR_CUDA_ERROR();

                    cudaGraphicsUnregisterResource(iResource);
                    cudaGraphicsUnregisterResource(vResource);
                    cudaGraphicsUnregisterResource(nResource);
                    cudaGraphicsUnregisterResource(cResource);
                    CHECK_FOR_CUDA_ERROR();
                }else{
                    // Geometry is still on the CPU
                    throw Exception("Not implemented");
                }
            }
            
            void GeometryList::AddScene(ISceneNode* node){
                currentModelMat = Matrix<4,4, float>();
                node->Accept(*this);
            }

            void GeometryList::VisitTransformationNode(TransformationNode* node){
                // push transformation matrix
                Matrix<4,4,float> m = node->GetTransformationMatrix();
                Matrix<4, 4, float> oldModelMat = currentModelMat;
                currentModelMat = m * currentModelMat;

                // traverse sub nodes
                node->VisitSubNodes(*this);

                // pop transformation matrix
                currentModelMat = oldModelMat;
            }
            
            void GeometryList::VisitMeshNode(MeshNode* node){
                AddMesh(node->GetMesh(), currentModelMat);

                node->VisitSubNodes(*this);
            }

        }
    }
}
