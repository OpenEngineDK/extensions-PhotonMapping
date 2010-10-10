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
#include <Scene/TriangleUpperNode.h>
#include <Scene/TriangleLowerNode.h>
#include <Utils/CUDA/GeometryList.h>
#include <Utils/CUDA/Segments.h>

#include <Meta/CUDPP.h>

namespace OpenEngine {
    namespace Scene {
        class ISceneNode;
    }
    namespace Utils {
        namespace CUDA {

            class TriangleMap {
            public:
                Scene::ISceneNode* scene;
                GeometryList* geom;

                Scene::TriangleUpperNode* upperNodes;
                Scene::TriangleLowerNode* lowerNodes;

                int triangles;
                CUDPPConfiguration scanConfig;
                CUDPPHandle scanHandle;

                CUDADataBlock<1, point> *tempAabbMin;
                CUDADataBlock<1, point> *tempAabbMax;

                Segments segments;
                CUDADataBlock<1, int> *nodeSegments;
                
            public:
                TriangleMap(Scene::ISceneNode* scene);

                void Create();
                void Setup();

                void ProcessUpperNodes(int activeIndex, int activeRange, 
                                       int &leafsCreated, int &childrenCreated);

                void Segment(int activeIndex, int activeRange);
            };

        }
    }
}

#endif
