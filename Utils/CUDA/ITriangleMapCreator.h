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

#include <Scene/TriangleNode.h>
#include <Resources/CUDA/CUDADataBlock.h>

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {

            class ITriangleMapCreator {
            private:
                //int leafs;
                Resources::CUDA::CUDADataBlock<1, float4>* primMin;
                Resources::CUDA::CUDADataBlock<1, float4>* primMax;

                Resources::CUDA::CUDADataBlock<1, int>* leafIDs;

            public:
                ITriangleMapCreator() : leafIDs(NULL) {}
                virtual ~ITriangleMapCreator() { if (leafIDs) delete leafIDs;}

                virtual void Create(Scene::TriangleNode* nodes, 
                                    int activeIndex, int activeRange,
                                    ITriangleMapCreator* upper);
                
                Resources::CUDA::CUDADataBlock<1, int>* GetLeafIDs() { return leafIDs; }
            };

        }
    }
}

#endif
