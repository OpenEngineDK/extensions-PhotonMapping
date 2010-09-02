// Variables for doing a split
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Meta/CUDA.h>
#include <Utils/CUDA/Types.h>
#include <string>

#ifndef _SPLIT_VAR_H_
#define _SPLIT_VAR_H_

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {
            
            class SplitVar {
            public:
                unsigned int *side; // can be replaced by bit fiddling at some point to save
                // memory.
                unsigned int *prefixSum;     
                unsigned int totalF; // Sum of false's in f
                //unsigned int *left; // move left addresses
                //unsigned int *right; // move right addresses
                //unsigned int *addr; // actual addresses
                // the 3 above arrays should be superflues and can just be stored
                // as variables inside the threads.

                // Temp values for photon splitting
                point *tempPos;

                unsigned int size;
                
            public:
                void Init(unsigned int s){
                    size = s;
                    cudaMalloc(&side, s * sizeof(unsigned int));
                    cudaMalloc(&prefixSum, s * sizeof(unsigned int));

                    cudaMalloc(&tempPos, s * sizeof(point));
                }

                std::string SideToString(unsigned int begin, unsigned int end);
                std::string PrefixSumToString(unsigned int begin, unsigned int end);
            };
        }
    }
}

#endif // _SPLIT_VAR_H_
