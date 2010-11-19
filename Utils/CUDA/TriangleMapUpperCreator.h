// Triangle map upper node creator interface for CUDA
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _TRIANGLE_MAP_UPPER_CREATOR_H_
#define _TRIANGLE_MAP_UPPER_CREATOR_H_

#include <Utils/CUDA/ITriangleMapCreator.h>

#include <Utils/CUDA/Segments.h>

#include <Meta/CUDPP.h>

namespace OpenEngine {    
    namespace Scene {
        class TriangleNode;
    }
    namespace Utils {
        namespace CUDA {

            class TriangleMapUpperCreator : public ITriangleMapCreator {
            protected:
                unsigned int timerID;

                TriangleMap* map;
                Scene::TriangleNode* nodes;

                Segments segments;
                Resources::CUDA::CUDADataBlock<1, int> *nodeSegments;

                // Primitive aabb values
                Resources::CUDA::CUDADataBlock<1, float4> *aabbMin;
                Resources::CUDA::CUDADataBlock<1, float4> *aabbMax;

                // Arrays for holding reduced aabb values
                Resources::CUDA::CUDADataBlock<1, float4> *tempAabbMin;
                Resources::CUDA::CUDADataBlock<1, float4> *tempAabbMax;

                // Split variables
                Resources::CUDA::CUDADataBlock<1, int> *splitSide;
                Resources::CUDA::CUDADataBlock<1, int> *splitAddr;
                Resources::CUDA::CUDADataBlock<1, int> *leafSide;
                Resources::CUDA::CUDADataBlock<1, int> *leafAddr;

                int upperLeafPrimitives;

                CUDPPConfiguration scanConfig;
                CUDPPHandle scanHandle;
                int scanSize;

                CUDPPConfiguration scanInclConfig;
                CUDPPHandle scanInclHandle;
                int scanInclSize;

            public:
                TriangleMapUpperCreator() {}
                virtual ~TriangleMapUpperCreator() {}

                virtual void Create(TriangleMap* map, 
                                    Resources::CUDA::CUDADataBlock<1, int>* upperLeafIDs) {}

                void ProcessNodes(int activeIndex, int activeRange, 
                                  int &childrenCreated) {}

                void Segment(int activeIndex, int activeRange) {}
                
                void ReduceAabb(int activeIndex, int activeRange) {}
                
                void CreateChildren(int activeIndex, int activeRange,
                                    int &childrenCreated) {}

                void CheckUpperNode(int index, float4 aabbMin, float4 aabbMax, int activeRange = 0) {}
                void CheckUpperLeaf(int index, float4 aabbMin, float4 aabbMax) {}
                void CheckSplits() {}

            };

        }
    }
}

#endif
