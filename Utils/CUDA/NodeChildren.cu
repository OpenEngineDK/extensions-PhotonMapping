// Temp variables for holding temp child node values.
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Utils/CUDA/NodeChildren.h>
#include <Utils/CUDA/Utils.h>

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {
            
            NodeChildren::NodeChildren(int size)
                : size(size) {
                cudaSafeMalloc(&photonInfo, size * sizeof(int2));
                cudaSafeMalloc(&parents, size * sizeof(int));
            }

            void NodeChildren::Resize(int size){
                cudaFree(photonInfo);
                cudaSafeMalloc(&photonInfo, size * sizeof(int2));
                cudaFree(parents);
                cudaSafeMalloc(&parents, size * sizeof(int));
            }

        }
    }
}
