// Transparent node.
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Scene/TransparentNode.h>

namespace OpenEngine {
    namespace Scene {

        TransparentNode::TransparentNode()
            : MeshNode() {}

        TransparentNode::TransparentNode(MeshPtr mesh, float refractionAngle)
            : MeshNode(mesh), refractionAngle(refractionAngle) {}

    }
}
