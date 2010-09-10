// Variables for doing an upper node split
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Meta/CUDA.h>
#include <string>

#ifndef _UPPER_SPLIT_VAR_H_
#define _UPPER_SPLIT_VAR_H_

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {
            
            class UpperSplitVar {
            public:
                unsigned int size;

                unsigned int *side;
                unsigned int *prefixSum;     
                unsigned int totalF; // Sum of children's in
                unsigned int *tempParent;
                unsigned int *tempIndex;
                unsigned int *tempRange;

            public:
                UpperSplitVar(unsigned int s);
            };

        }
    }
}

#endif
