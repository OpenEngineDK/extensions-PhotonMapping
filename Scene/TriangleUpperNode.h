// KD tree upper node for triangles
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _CUDA_TRIANGLE_UPPER_NODE_H_
#define _CUDA_TRIANGLE_UPPER_NODE_H_

#include <Scene/KDNode.h>

namespace OpenEngine {
    namespace Scene {

        class TriangleUpperNode : public KDNode {
        public:

        public:
            TriangleUpperNode();
            TriangleUpperNode(int size);

            void Resize(int i);
        };

    }
}

#endif