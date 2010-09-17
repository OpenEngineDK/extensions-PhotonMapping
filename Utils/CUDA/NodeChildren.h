// Temp variables for holding temp child node values.
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _NODE_CHILDREN_H_
#define _NODE_CHILDREN_H_

#include <Meta/CUDA.h>

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {
            
            class NodeChildren {
            public:
                int size;
                int2 *photonInfo;
                int *parents;

            public:
                NodeChildren()
                    : size(0), photonInfo(NULL), parents(NULL) {}
                NodeChildren(int size);

                void Resize(int size);
            };
            
        }
    }
}

#endif
