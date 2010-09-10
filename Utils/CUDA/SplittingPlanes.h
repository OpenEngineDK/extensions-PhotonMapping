// List of splitting planes.
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _SPLITTING_PLANES_H_
#define _SPLITTING_PLANES_H_

#include <Meta/CUDA.h>

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {

            class SplittingPlanes {
            public:
                // Each triangle set contains both the bitmap of the left and right side
                int2 *triangleSetX; // [left, right]
                int2 *triangleSetY; // [left, right]
                int2 *triangleSetZ; // [left, right]

                unsigned int size;
                

            public:
                SplittingPlanes();
                SplittingPlanes(unsigned int i);

                void Resize(unsigned int i);
            };

        }
    }
}

#endif
