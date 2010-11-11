// Triangle map creator interface for CUDA
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _I_TRIANGLE_MAP_CREATOR_H_
#define _I_TRIANGLE_MAP_CREATOR_H_

#include <Resources/CUDA/CUDADataBlock.h>

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {

            class TriangleMap;

            class ITriangleMapCreator {
            protected:
                Resources::CUDA::CUDADataBlock<1, float4>* primMin;
                Resources::CUDA::CUDADataBlock<1, float4>* primMax;

                Resources::CUDA::CUDADataBlock<1, int>* leafIDs;

            public:
                ITriangleMapCreator() : primMin(NULL), primMax(NULL), leafIDs(NULL) {}
                virtual ~ITriangleMapCreator() { if (leafIDs) delete leafIDs;}

                virtual void Create(TriangleMap* map,
                                    Resources::CUDA::CUDADataBlock<1, int>* upperLeafIDs);
                
                Resources::CUDA::CUDADataBlock<1, int>* GetLeafIDs() { return leafIDs; }
            };

        }
    }
}

#endif
