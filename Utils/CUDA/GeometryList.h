// Geometry List.
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _CUDA_GEOMETRY_CONTAINER_H_
#define _CUDA_GEOMETRY_CONTAINER_H_

#include <boost/shared_ptr.hpp>
#include <Math/Matrix.h>
#include <Meta/CUDA.h>
#include <Scene/ISceneNodeVisitor.h>
#include <Resources/CUDA/CUDADataBlock.h>

#include <string>

namespace OpenEngine {
    namespace Geometry {
        class Mesh;
        typedef boost::shared_ptr<Mesh> MeshPtr;
    }
    namespace Scene {
        class ISceneNode;
        class MeshNode;
        class TransformationNode;
    }
    namespace Utils {
        namespace CUDA {
            
            class GeometryList : public virtual Scene::ISceneNodeVisitor {
            public:
                unsigned int timerID;
                
                int maxSize, size;

                Resources::CUDA::CUDADataBlock<1, float4> *p0, *p1, *p2;
                Resources::CUDA::CUDADataBlock<1, float4> *n0, *n1, *n2;
                Resources::CUDA::CUDADataBlock<1, uchar4> *c0, *c1, *c2;

                // Visitor variables
                Math::Matrix<4,4, float> currentModelMat;

            public:
                GeometryList();
                GeometryList(int size);

                int GetSize() const { return size; }

                void Resize(int i);
                void Extend(int i);

                float4* GetP0Data() const { return p0->GetDeviceData(); }
                float4* GetP1Data() const { return p1->GetDeviceData(); }
                float4* GetP2Data() const { return p2->GetDeviceData(); }

                float4* GetNormal0Data() const { return n0->GetDeviceData(); }
                float4* GetNormal1Data() const { return n1->GetDeviceData(); }
                float4* GetNormal2Data() const { return n2->GetDeviceData(); }

                uchar4* GetColor0Data() const { return c0->GetDeviceData(); }
                uchar4* GetColor1Data() const { return c1->GetDeviceData(); }
                uchar4* GetColor2Data() const { return c2->GetDeviceData(); }
                
                std::string ToString(unsigned int i) const;

                void AddMesh(Geometry::MeshPtr mesh, 
                             Math::Matrix<4, 4, float> ModelView);

                void CollectGeometry(Scene::ISceneNode* node);

                void VisitRenderStateNode(Scene::RenderStateNode* node);
                void VisitTransformationNode(Scene::TransformationNode* node);
                void VisitMeshNode(Scene::MeshNode* node);
                
            };
            
        }
    }
}
    
#endif
