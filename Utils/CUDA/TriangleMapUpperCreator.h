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

#include <Scene/TriangleNode.h>
#include <Utils/CUDA/Segments.h>
#include <Utils/CUDA/TriangleMap.h>

#include <cudpp/cudpp.h>

//#define CPU_VERIFY true

namespace OpenEngine {    
    namespace Utils {
        namespace CUDA {

            class TriangleMapUpperCreator : public ITriangleMapCreator {
            protected:
                unsigned int timerID;

                TriangleMap* map;

                Segments segments;
                Resources::CUDA::CUDADataBlock<1, int> *nodeSegments;

                bool emptySpaceSplitting;
                static float emptySpaceThreshold;

                TriangleMap::SplitMethod splitMethod;

                // Primitive aabb values
                Resources::CUDA::CUDADataBlock<1, float4> *aabbMin;
                Resources::CUDA::CUDADataBlock<1, float4> *aabbMax;

                // Arrays for holding reduced aabb values
                Resources::CUDA::CUDADataBlock<1, float4> *tempAabbMin;
                Resources::CUDA::CUDADataBlock<1, float4> *tempAabbMax;

                // @OPT Perhaps use an indice array ie nextList,
                // instead of rearranging all the nodes? Should save
                // on cudpp scan invokations

                // Split variables
                Resources::CUDA::CUDADataBlock<1, int> *splitSide;
                Resources::CUDA::CUDADataBlock<1, int> *splitAddr;
                Resources::CUDA::CUDADataBlock<1, int> *leafSide;
                Resources::CUDA::CUDADataBlock<1, int> *leafAddr;
                Resources::CUDA::CUDADataBlock<1, char> *emptySpacePlanes;
                Resources::CUDA::CUDADataBlock<1, int> *emptySpaceNodes; // Can be replaced by splitSide? Probably
                Resources::CUDA::CUDADataBlock<1, int> *emptySpaceAddrs;
                Resources::CUDA::CUDADataBlock<1, int> *nodeIndices;
                Resources::CUDA::CUDADataBlock<1, int2> *childSize;
                Resources::CUDA::CUDADataBlock<1, Scene::KDNode::amount> *tempNodeAmount;

                int upperLeafPrimitives;

                CUDPPConfiguration scanConfig;
                CUDPPHandle scanHandle;
                int scanSize;

                CUDPPConfiguration scanInclConfig;
                CUDPPHandle scanInclHandle;
                int scanInclSize;

            public:
                TriangleMapUpperCreator();
                virtual ~TriangleMapUpperCreator();

                virtual void Create(TriangleMap* map, 
                                    Resources::CUDA::CUDADataBlock<1, int>* upperLeafIDs);

                inline void SplitEmptySpace(const bool s) { emptySpaceSplitting = s; }
                inline bool IsSplittingEmptySpace() const { return emptySpaceSplitting; }
                inline void SetSplitMethod(const TriangleMap::SplitMethod s) { splitMethod = s; }
                inline TriangleMap::SplitMethod GetSplitMethod() const { return splitMethod; }
                static void SetEmptySpaceThreshold(const float t) { emptySpaceThreshold = t; }
                static float GetEmptySpaceThreshold() { return emptySpaceThreshold; }
                
                void ProcessNodes(int activeIndex, int activeRange, 
                                  int &childrenCreated);

            protected:
                void Segment(int activeIndex, int activeRange);
                
                void ReduceAabb(int &activeIndex, int activeRange);

                void CreateEmptySplits(int &activeIndex, int activeRange);
                
                void CreateChildren(int activeIndex, int activeRange,
                                    int &childrenCreated);

                void CheckSegmentReduction(int activeIndex, int activeRange,
                                           Segments &segments, 
                                           float4 **finalMin, 
                                           float4 **finalMax);

                void CheckFinalReduction(int activeIndex, int activeRange,
                                         Scene::TriangleNode* nodes, 
                                         float4 *finalMin, 
                                         float4 *finalMax);

                void CheckPrimAabb(Resources::CUDA::CUDADataBlock<1, float4> *aabbMin, 
                                   Resources::CUDA::CUDADataBlock<1, float4> *aabbMax);
                void CheckUpperNode(int index, float4 aabbMin, float4 aabbMax, int activeRange = 0);
                void CheckUpperLeaf(int index, float4 aabbMin, float4 aabbMax);
                void CheckSplits();

            };

        }
    }
}

#endif
