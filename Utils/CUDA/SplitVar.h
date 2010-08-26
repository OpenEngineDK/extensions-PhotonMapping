// Variables for doing a split
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Meta/CUDA.h>

#ifndef _SPLIT_VAR_H_
#define _SPLIT_VAR_H_

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {
            
            class SplitVar {
            public:
                bool *side; // can be replaced by bit fiddling at some point to save
                // memory.
                unsigned int *prefixSum;     
                unsigned int totalF; // Sum of false's in f
                //unsigned int *left; // move left addresses
                //unsigned int *right; // move right addresses
                unsigned int *adress; // actual addresses
                // the 3 above arrays should be superflues and can just be stored
                // as variables inside the threads.
                float3 *tempPos;
                unsigned int size;
                
            public:
                void Init(unsigned int s){
                    size = s;
                    cudaMalloc(&side, s * sizeof(bool));
                    cudaMalloc(&prefixSum, s * sizeof(unsigned int));
                    cudaMalloc(&tempPos, s * sizeof(float3));
                }
            };
        }
    }
}

#endif // _SPLIT_VAR_H_
