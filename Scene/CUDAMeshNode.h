// CUDA Mesh Node.
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _CUDA_MESH_NODE_H_
#define _CUDA_MESH_NODE_H_

#include <Scene/ISceneNode.h>
#include <Resources/CUDA/CUDADataBlock.h>

namespace OpenEngine {
    namespace Scene {
        class MeshNode;
        
        class CUDAMeshNode : public ISceneNode {
            OE_SCENE_NODE(CUDAMeshNode, ISceneNode)
        protected:
            Resources::CUDA::CUDADataBlock<1, float4> *vertices;
            Resources::CUDA::CUDADataBlock<1, float4> *normals;
            Resources::CUDA::CUDADataBlock<1, uchar4> *colors;
            Resources::CUDA::CUDADataBlock<1, unsigned int> *indices;

        public:
            CUDAMeshNode() {}
            CUDAMeshNode(MeshNode* mesh);
            virtual ~CUDAMeshNode() {}

            float4* GetVertexData() const { return vertices->GetDeviceData(); }
            float4* GetNormalData() const { return normals->GetDeviceData(); }
            uchar4* GetColorData() const { return colors->GetDeviceData(); }
            unsigned int* GetIndexData() const { return indices->GetDeviceData(); }
            
            unsigned int GetSize() const { return indices->GetSize(); }
        };
            
    }
}

#endif
