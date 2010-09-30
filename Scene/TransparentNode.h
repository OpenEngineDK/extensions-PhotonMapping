// Transparent node.
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _TRANSPARENT_MESH_NODE_H_
#define _TRANSPARENT_MESH_NODE_H_

#include <Scene/MeshNode.h>

namespace OpenEngine {
    namespace Scene {

        class TransparentNode : public MeshNode {
        protected:
            float refractionAngle;
            
        public:
            TransparentNode();
            explicit TransparentNode(MeshPtr mesh, float refractionAngle);
        };

    }
}

#endif
