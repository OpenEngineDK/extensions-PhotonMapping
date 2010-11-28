// Triangle map balanced creator interface for CUDA
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _TRIANGLE_MAP_BALANCED_FAST_CREATOR_H_
#define _TRIANGLE_MAP_BALANCED_FAST_CREATOR_H_

#include <Utils/CUDA/ITriangleMapCreator.h>

#include <cudpp/cudpp.h>

namespace OpenEngine {    
    namespace Utils {
        namespace CUDA {

            class TriangleMapBalancedFast : public ITriangleMapCreator {
            protected:
                unsigned int timerID;

                Resources::CUDA::CUDADataBlock<1, int4> *splitTriangleSet;

                Resources::CUDA::CUDADataBlock<1, int> *indices;
                Resources::CUDA::CUDADataBlock<1, int> *newIndices;
                Resources::CUDA::CUDADataBlock<1, unsigned int> *validIndices;

                CUDPPConfiguration compactConfig;
                CUDPPHandle compactHandle;
                int compactSize;
                size_t *numValid;

            public:
                TriangleMapBalancedFast();
                virtual ~TriangleMapBalancedFast();

                virtual void Create(TriangleMap* map, 
                                    Resources::CUDA::CUDADataBlock<1, int>* upperLeafIDs);
                
                void PreprocessLowerNodes(int activeIndex, int activeRange, 
                                          TriangleMap* map, Resources::CUDA::CUDADataBlock<1, int>* upperLeafIDs);
                
                void ProcessLowerNodes(int activeIndex, int activeRange, 
                                       TriangleMap* map, int &childrenCreated);

            };
            
        }
    }
}

#endif
