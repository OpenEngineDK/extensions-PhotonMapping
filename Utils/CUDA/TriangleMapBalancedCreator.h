// Triangle map balanced creator interface for CUDA
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _TRIANGLE_MAP_BALANCED_CREATOR_H_
#define _TRIANGLE_MAP_BALANCED_CREATOR_H_

#include <Utils/CUDA/ITriangleMapCreator.h>

#include <Scene/KDNode.h>

#include <cudpp/cudpp.h>

namespace OpenEngine {    
    namespace Utils {
        namespace CUDA {

            class TriangleMapBalancedCreator : public ITriangleMapCreator {
            protected:
                unsigned int timerID;

                bool removeFalsePrimitives;
                float traversalCost;

                Resources::CUDA::CUDADataBlock<Scene::KDNode::bitmap4> *splitTriangleSet;

                Resources::CUDA::CUDADataBlock<Scene::KDNode::bitmap2>* childSets;
                Resources::CUDA::CUDADataBlock<int>* splitSide;
                Resources::CUDA::CUDADataBlock<int>* splitAddr;

                CUDPPConfiguration scanConfig;
                CUDPPHandle scanHandle;
                int scanSize;

            public:
                TriangleMapBalancedCreator();
                virtual ~TriangleMapBalancedCreator();

                virtual void Create(TriangleMap* map, 
                                    Resources::CUDA::CUDADataBlock<int>* upperLeafIDs);
                
                void PreprocessLowerNodes(int activeIndex, int activeRange, 
                                          TriangleMap* map, Resources::CUDA::CUDADataBlock<int>* upperLeafIDs);
                
                void ProcessLowerNodes(int activeIndex, int activeRange, 
                                       TriangleMap* map, Resources::CUDA::CUDADataBlock<int>* upperLeafIDs,
                                       int &childrenCreated);

                void SetTraversalCost(const float t) { traversalCost = t; }

            };
            
        }
    }
}

#endif
