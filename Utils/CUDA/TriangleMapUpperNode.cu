// Triangle map class for CUDA
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Utils/CUDA/TriangleMap.h>

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {

            void TriangleMap::ProcessUpperNodes(int activeIndex, int activeRange, 
                                                int &leafsCreated, int &childrenCreated){
                leafsCreated = 0;
                childrenCreated = 0;
            }

        }
    }
}
