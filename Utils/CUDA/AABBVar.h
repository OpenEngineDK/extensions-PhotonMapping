// Variables for calculating Axis Aligned Bounding Boxes
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Meta/CUDA.h>
#include <string>
#include <Utils/CUDA/Convert.h>
#include <Utils/CUDA/Types.h>

#ifndef _AABB_VAR_H_
#define _AABB_VAR_H_

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {
            
            class AABBVar {
            public:
                unsigned int size;
                point *max;
                point *min;
                unsigned int *owner;

            public:
                AABBVar()
                    : size(0), max(NULL), min(NULL), owner(NULL) {}

                AABBVar(unsigned int size)
                    : size(size) {
                    cudaMalloc(&max, size * sizeof(point));
                    cudaMalloc(&min, size * sizeof(point));
                    cudaMalloc(&owner, size * sizeof(unsigned int));
                    CHECK_FOR_CUDA_ERROR();
                }

                
                std::string MaxToString(unsigned int size = 64);
                std::string MinToString(unsigned int size = 64);
            };
        }
    }
}

#endif //_AABB_VAR_H_
