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
#include <Math/CUDA/Matrix.h>
#include <Scene/ISceneNode.h>
#include <Scene/MeshNode.h>
#include <Scene/CUDAMeshNode.h>
#include <Scene/RenderStateNode.h>
#include <Scene/TransformationNode.h>
#include <Utils/CUDA/Utils.h>
#include <Utils/CUDA/Convert.h>

#include <sstream>

using namespace OpenEngine::Geometry;
using namespace OpenEngine::Math;
using namespace OpenEngine::Math::CUDA;
using namespace OpenEngine::Scene;
using namespace OpenEngine::Resources::CUDA;

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {

            GeometryList::GeometryList()
                : maxSize(0), size(0) {}

            GeometryList::GeometryList(int size)
                : maxSize(size), size(0) {

                cutCreateTimer(&timerID);

                p0 = new CUDADataBlock<1, float4>(maxSize);
                p1 = new CUDADataBlock<1, float4>(maxSize);
                p2 = new CUDADataBlock<1, float4>(maxSize);

                n0 = new CUDADataBlock<1, float4>(maxSize);
                n1 = new CUDADataBlock<1, float4>(maxSize);
                n2 = new CUDADataBlock<1, float4>(maxSize);

                c0 = new CUDADataBlock<1, uchar4>(maxSize);
                c1 = new CUDADataBlock<1, uchar4>(maxSize);
                c2 = new CUDADataBlock<1, uchar4>(maxSize);
            }

            void GeometryList::Resize(int i){
                p0->Resize(i); p1->Resize(i); p2->Resize(i);
                n0->Resize(i); n1->Resize(i); n2->Resize(i);
                c0->Resize(i); c1->Resize(i); c2->Resize(i);

                maxSize = i;
                size = min(size, i);
            }

            void GeometryList::Extend(int i){
                if (maxSize < i)
                    Resize(i);
            }

            std::string GeometryList::ToString(unsigned int i) const {
                std::ostringstream out;

                out <<  "Triangle #" << i << "\n";

                float4 h_p0, h_p1, h_p2;
                cudaMemcpy(&h_p0, p0->GetDeviceData() + i, sizeof(float4), cudaMemcpyDeviceToHost);
                cudaMemcpy(&h_p1, p1->GetDeviceData() + i, sizeof(float4), cudaMemcpyDeviceToHost);
                cudaMemcpy(&h_p2, p2->GetDeviceData() + i, sizeof(float4), cudaMemcpyDeviceToHost);
                out << "Points: " << Utils::CUDA::Convert::ToString(h_p0) << ", " << Utils::CUDA::Convert::ToString(h_p1) << " & " << Utils::CUDA::Convert::ToString(h_p2) << "\n";

                float4 h_n0, h_n1, h_n2;
                cudaMemcpy(&h_n0, n0->GetDeviceData() + i, sizeof(float4), cudaMemcpyDeviceToHost);
                cudaMemcpy(&h_n1, n1->GetDeviceData() + i, sizeof(float4), cudaMemcpyDeviceToHost);
                cudaMemcpy(&h_n2, n2->GetDeviceData() + i, sizeof(float4), cudaMemcpyDeviceToHost);
                out << "Normals: " << Utils::CUDA::Convert::ToString(h_n0) << ", " << Utils::CUDA::Convert::ToString(h_n1) << " & " << Utils::CUDA::Convert::ToString(h_n2) << "\n";

                uchar4 h_c0, h_c1, h_c2;
                cudaMemcpy(&h_c0, c0->GetDeviceData() + i, sizeof(uchar4), cudaMemcpyDeviceToHost);
                cudaMemcpy(&h_c1, c1->GetDeviceData() + i, sizeof(uchar4), cudaMemcpyDeviceToHost);
                cudaMemcpy(&h_c2, c2->GetDeviceData() + i, sizeof(uchar4), cudaMemcpyDeviceToHost);
                out << "Colors: " << Utils::CUDA::Convert::ToString(h_c0) << ", " << Utils::CUDA::Convert::ToString(h_c1) << " & " << Utils::CUDA::Convert::ToString(h_c2) << "\n";
                
                return out.str();
            }

            __global__ void AddMeshKernel(unsigned int *indices,
                                          float3 *verticesIn,
                                          float3 *normalsIn,
                                          float4 *colorsIn,
                                          const Matrix44f modelMat, const Matrix33f normalMat,
                                          float4 *p0, float4 *p1, float4 *p2,
                                          float4 *n0, float4 *n1, float4 *n2,
                                          uchar4 *c0, uchar4 *c1, uchar4 *c2,
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

            __global__ void AddMeshKernel(unsigned int *indices,
                                          float4 *verticesIn,
                                          float4 *normalsIn,
                                          uchar4 *colorsIn,
                                          const Matrix44f modelMat, const Matrix33f normalMat,
                                          float4 *p0, float4 *p1, float4 *p2,
                                          float4 *n0, float4 *n1, float4 *n2,
                                          uchar4 *c0, uchar4 *c1, uchar4 *c2,
                                          int size){
                
                const int id = blockDim.x * blockIdx.x + threadIdx.x;

                if (id < size){
                    const int i = __mul24(id, 3);
                    const unsigned int i0 = indices[i];
                    const unsigned int i1 = indices[i+1];
                    const unsigned int i2 = indices[i+2];
                    const float4 v0 = verticesIn[i0];
                    const float4 v1 = verticesIn[i1];
                    const float4 v2 = verticesIn[i2];
                    
                    p0[id] = modelMat * v0;
                    p1[id] = modelMat * v1;
                    p2[id] = modelMat * v2;
                    
                    n0[id] = make_float4(normalMat * make_float3(normalsIn[i0]), 0);
                    n1[id] = make_float4(normalMat * make_float3(normalsIn[i1]), 0);
                    n2[id] = make_float4(normalMat * make_float3(normalsIn[i2]), 0);

                    c0[id] = colorsIn[i0];
                    c1[id] = colorsIn[i1];
                    c2[id] = colorsIn[i2];
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

                    START_TIMER(timerID);
                    unsigned int triangles = indices->GetSize() / 3;
                    Extend(size + triangles);
                    
                    cudaGraphicsResource *iResource, *vResource, *nResource, *cResource;
                    cudaGraphicsGLRegisterBuffer(&iResource, indices->GetID(), cudaGraphicsMapFlagsReadOnly);
                    cudaGraphicsMapResources(1, &iResource, 0);
                    CHECK_FOR_CUDA_ERROR();
                    cudaGraphicsGLRegisterBuffer(&vResource, vertices->GetID(), cudaGraphicsMapFlagsReadOnly);
                    cudaGraphicsMapResources(1, &vResource, 0);
                    CHECK_FOR_CUDA_ERROR();
                    cudaGraphicsGLRegisterBuffer(&nResource, normals->GetID(), cudaGraphicsMapFlagsReadOnly);
                    cudaGraphicsMapResources(1, &nResource, 0);
                    CHECK_FOR_CUDA_ERROR();
                    cudaGraphicsGLRegisterBuffer(&cResource, colors->GetID(), cudaGraphicsMapFlagsReadOnly);
                    cudaGraphicsMapResources(1, &cResource, 0);
                    CHECK_FOR_CUDA_ERROR();
                    
                    size_t bytes;
                    unsigned int* in;
                    cudaGraphicsResourceGetMappedPointer((void**)&in, &bytes,
                                                         iResource);
                    CHECK_FOR_CUDA_ERROR();
                    float3* pos;
                    cudaGraphicsResourceGetMappedPointer((void**)&pos, &bytes,
                                                         vResource);
                    CHECK_FOR_CUDA_ERROR();
                    float3* norms;
                    cudaGraphicsResourceGetMappedPointer((void**)&norms, &bytes,
                                                         nResource);
                    CHECK_FOR_CUDA_ERROR();
                    float4* cols;
                    cudaGraphicsResourceGetMappedPointer((void**)&cols, &bytes,
                                                         cResource);
                    CHECK_FOR_CUDA_ERROR();

                    unsigned int blocks, threads;
                    Calc1DKernelDimensions(indices->GetSize(), blocks, threads);
                    Math::CUDA::Matrix44f mat;
                    mat.Init(modelMat.GetTranspose());
                    Math::CUDA::Matrix33f normMat; // should be transposed and inverted, jada jada bla bla just don't do weird scaling
                    normMat.Init(mat);
                    CHECK_FOR_CUDA_ERROR();

                    AddMeshKernel<<<blocks, threads>>>(in, pos, norms, cols,
                                                       mat, normMat,
                                                       p0->GetDeviceData() + size, p1->GetDeviceData() + size, p2->GetDeviceData() + size,
                                                       n0->GetDeviceData() + size, n1->GetDeviceData() + size, n2->GetDeviceData() + size,
                                                       c0->GetDeviceData() + size, c1->GetDeviceData() + size, c2->GetDeviceData() + size,
                                                       triangles);
                    CHECK_FOR_CUDA_ERROR();

                    size += triangles;

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

                    PRINT_TIMER(timerID, "Geometry collection ");
                }else{
                    // Geometry is still on the CPU
                    throw Exception("Not implemented");
                }
            }

            void GeometryList::AddMesh(CUDAMeshNode* mesh, 
                                       Matrix<4, 4, float> modelMat){
                
                //START_TIMER(timerID);

                unsigned int triangles = mesh->GetSize() / 3;
                Extend(size + triangles);                

                unsigned int blocks, threads;
                Calc1DKernelDimensions(mesh->GetSize(), blocks, threads);
                Math::CUDA::Matrix44f mat;
                mat.Init(modelMat.GetTranspose());
                Math::CUDA::Matrix33f normMat; // should be transposed and inverted, jada jada bla bla just don't do weird scaling
                normMat.Init(mat);
                CHECK_FOR_CUDA_ERROR();
                
                AddMeshKernel<<<blocks, threads>>>(mesh->GetIndexData(), mesh->GetVertexData(), mesh->GetNormalData(), mesh->GetColorData(),
                                                   mat, normMat,
                                                   p0->GetDeviceData() + size, p1->GetDeviceData() + size, p2->GetDeviceData() + size,
                                                   n0->GetDeviceData() + size, n1->GetDeviceData() + size, n2->GetDeviceData() + size,
                                                   c0->GetDeviceData() + size, c1->GetDeviceData() + size, c2->GetDeviceData() + size,
                                                   triangles);
                CHECK_FOR_CUDA_ERROR();

                

                size += triangles;
                
                //PRINT_TIMER(timerID, "Geometry collection ");
            }
            
            void GeometryList::CollectGeometry(ISceneNode* node){
                currentModelMat = Matrix<4,4, float>();
                size = 0;
                node->Accept(*this);
            }

            void GeometryList::VisitRenderStateNode(RenderStateNode* node){
                node->VisitSubNodes(*this);
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
                if (node->GetMesh()->GetGeometrySet()->GetVertices()->GetID() != 0){
                    AddMesh(node->GetMesh(), currentModelMat);
                    
                    node->VisitSubNodes(*this);
                }else{
                    CUDAMeshNode* mesh = new CUDAMeshNode(node);

                    node->GetParent()->ReplaceNode(node, mesh);

                    std::list<ISceneNode*> subNodes = node->subNodes;
                    for (std::list<ISceneNode*>::iterator itr = subNodes.begin();
                         itr != subNodes.end(); ++itr){
                        node->RemoveNode(*itr);
                        mesh->AddNode(*itr);
                    }

                    mesh->Accept(*this);
                }
            }

            void GeometryList::VisitCUDAMeshNode(CUDAMeshNode* node){
                AddMesh(node, currentModelMat);

                node->VisitSubNodes(*this);
            }

        }
    }
}
