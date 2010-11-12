// Triangle map class for CUDA
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _TRIANGLE_MAP_H_
#define _TRIANGLE_MAP_H_

#include <Utils/CUDA/GeometryList.h>
#include <Utils/CUDA/GeometryList.h>
#include <Utils/CUDA/Segments.h>

#include <Meta/CUDPP.h>

#define CPU_VERIFY true

namespace OpenEngine {
    namespace Scene {
        class ISceneNode;
        class TriangleNode;
    }
    namespace Utils {
        namespace CUDA {

            class ITriangleMapCreator;

            class TriangleMap {
            public:
                unsigned int timerID;

                Scene::ISceneNode* scene;
                GeometryList* geom;

                Scene::TriangleNode* nodes;

                int triangles;

                ITriangleMapCreator* lowerCreator;
                
                float emptySpaceThreshold;

                CUDPPConfiguration scanConfig;
                CUDPPHandle scanHandle;
                int scanSize;

                CUDPPConfiguration scanInclConfig;
                CUDPPHandle scanInclHandle;
                int scanInclSize;

                CUDADataBlock<1, float4> *aabbMin;
                CUDADataBlock<1, float4> *aabbMax;
                CUDADataBlock<1, float4> *tempAabbMin;
                CUDADataBlock<1, float4> *tempAabbMax;
                CUDADataBlock<1, float4> *primMin;
                CUDADataBlock<1, float4> *primMax;
                CUDADataBlock<1, int> *primIndices;

                Segments segments;
                CUDADataBlock<1, int> *nodeSegments;

                // Split vars
                CUDADataBlock<1, int> *splitSide;
                CUDADataBlock<1, int> *splitAddr;
                CUDADataBlock<1, int> *leafSide;
                CUDADataBlock<1, int> *leafAddr;
                CUDADataBlock<1, int> *emptySpaceSplits;
                CUDADataBlock<1, int> *emptySpaceAddrs;
                CUDADataBlock<1, int2> *childSize;
                int upperLeafPrimitives;

                CUDADataBlock<1, int> *leafIDs;
                
            public:
                TriangleMap(Scene::ISceneNode* scene);

                void Create();
                void Setup();

                GeometryList* GetGeometry() const { return geom; }
                Scene::TriangleNode* GetNodes() const { return nodes; }
                CUDADataBlock<1, int>* GetPrimitiveIndices() const { return primIndices; }
                
                void CreateUpperNodes();

                void ProcessUpperNodes(int activeIndex, int activeRange, 
                                       int &childrenCreated);

                void Segment(int activeIndex, int activeRange);

                void ReduceAabb(int activeIndex, int activeRange);

                void CreateChildren(int activeIndex, int activeRange,
                                    int &childrenCreated);

                void CheckUpperNode(int index, float4 aabbMin, float4 aabbMax, int activeRange = 0);
                void CheckUpperLeaf(int index, float4 aabbMin, float4 aabbMax);
                void CheckSplits();

            };

        }
    }
}

#endif
