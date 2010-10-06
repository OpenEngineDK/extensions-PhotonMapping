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
            protected:
                unsigned int maxSize;

                Resources::CUDA::CUDADataBlock<1, float4> *p0, *p1, *p2;
                Resources::CUDA::CUDADataBlock<1, float4> *n0, *n1, *n2;
                Resources::CUDA::CUDADataBlock<1, uchar4> *c0, *c1, *c2;
                Resources::CUDA::CUDADataBlock<1, float4> *aabbMin, *aabbMax;
                Resources::CUDA::CUDADataBlock<1, float> *surfaceArea;

                // Visitor variables
                Math::Matrix<4,4, float> currentModelMat;

            public:
                GeometryList();
                GeometryList(int size);

                void Resize(int size);

                void AddMesh(Geometry::MeshPtr mesh, 
                             Math::Matrix<4, 4, float> ModelView);

                void AddScene(Scene::ISceneNode* node);

                void VisitTransformationNode(Scene::TransformationNode* node);
                void VisitMeshNode(Scene::MeshNode* node);
                
            };
            
        }
    }
}
    
#endif
