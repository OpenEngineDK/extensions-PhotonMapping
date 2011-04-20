// Oscillating CUDA Mesh Node.
// -------------------------------------------------------------------
// Copyright (C) 2011 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _OSCILATING_CUDA_MESH_NODE_H_
#define _OSCILATING_CUDA_MESH_NODE_H_

#include <Scene/CUDAMeshNode.h>
#include <Core/EngineEvents.h>

namespace OpenEngine {
    namespace Scene {

        class OscCUDAMeshNode : public CUDAMeshNode,
                                public IListener<Core::ProcessEventArg> {
            OE_SCENE_NODE(OscCUDAMeshNode, CUDAMeshNode)
        protected:
            float width, depth, height;
            int detail;
            unsigned int time;
            
        public:
            OscCUDAMeshNode() : CUDAMeshNode() {}
            OscCUDAMeshNode(float width, float depth, float height, int detail);
            virtual ~OscCUDAMeshNode() {}

            void Init();

            void Handle(Core::ProcessEventArg arg);
            
        };

    }
}

#endif
