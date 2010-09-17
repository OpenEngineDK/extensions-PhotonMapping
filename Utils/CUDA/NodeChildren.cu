// Temp variables for holding temp child node values.
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Utils/CUDA/NodeChildren.h>

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {
            
            NodeChildren::NodeChildren(int size)
                : size(size) {
                cudaMalloc(&photonInfo, size * sizeof(int2));
                cudaMalloc(&parents, size * sizeof(int));
            }

            void NodeChildren::Resize(int size){
                cudaFree(photonInfo);
                cudaMalloc(&photonInfo, size * sizeof(int2));
                cudaFree(parents);
                cudaMalloc(&parents, size * sizeof(int));
            }

        }
    }
}
