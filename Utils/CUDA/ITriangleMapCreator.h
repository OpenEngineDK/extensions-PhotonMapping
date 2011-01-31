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
                Resources::CUDA::CUDADataBlock<1, int>* primIndices;

                Resources::CUDA::CUDADataBlock<1, int>* leafIDs;

                bool propagateAabbs;

            public:
                ITriangleMapCreator() : primMin(NULL), primMax(NULL), primIndices(NULL), leafIDs(NULL), propagateAabbs(true) {}
                virtual ~ITriangleMapCreator() { 
                    if (primMin) delete primMin;
                    if (primMax) delete primMax;
                    if (primIndices) delete primIndices;

                    if (leafIDs) delete leafIDs; 
                }

                /**
                 * Creates more nodes in the triangle map from the previous leaf nodes.
                 * Is responsible for updating the triangle map's datablocks afterwards.
                 */
                virtual void Create(TriangleMap* map,
                                    Resources::CUDA::CUDADataBlock<1, int>* upperLeafIDs) = 0;
                
                Resources::CUDA::CUDADataBlock<1, float4>* GetPrimMin() { return primMin; }
                Resources::CUDA::CUDADataBlock<1, float4>* GetPrimMax() { return primMax; }
                Resources::CUDA::CUDADataBlock<1, int>* GetPrimIndices() { return primIndices; }
                Resources::CUDA::CUDADataBlock<1, int>* GetLeafIDs() { return leafIDs; }

                void SetPropagateBoundingBox(const bool p) {propagateAabbs = p; }
                bool GetPropagateBoundingBox() const { return propagateAabbs; }
            };

        }
    }
}

#endif
