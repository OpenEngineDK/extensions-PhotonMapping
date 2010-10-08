// KD tree upper node for triangles
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Scene/TriangleUpperNode.h>

namespace OpenEngine {
    namespace Scene {
        
        TriangleUpperNode::TriangleUpperNode()
            : KDNode() {}

        TriangleUpperNode::TriangleUpperNode(int size)
            : KDNode(size) {}

        void TriangleUpperNode::Resize(int i){
            KDNode::Resize(i);
        }

    }
}
