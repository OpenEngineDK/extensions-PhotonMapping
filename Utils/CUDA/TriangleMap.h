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

#include <Resources/CUDA/CUDADataBlock.h>
#include <Utils/CUDA/GeometryList.h>

//#define CPU_VERIFY true

namespace OpenEngine {
    namespace Scene {
        class ISceneNode;
        class TriangleNode;
    }
    namespace Utils {
        namespace CUDA {

            class ITriangleMapCreator;
            class TriangleMapUpperCreator;
            class TriangleMapBalancedCreator;            
            class TriangleMapSAHCreator;            

            class TriangleMap {
            public:
                enum LowerAlgorithm {BITMAP, BALANCED, SAH};
                enum SplitMethod {BOX, DIVIDE, SPLIT};
            public: //protected:
                unsigned int timerID;
                float constructionTime;

                Scene::ISceneNode* scene;
                GeometryList* geom;
                Scene::TriangleNode* nodes;

                TriangleMapUpperCreator* upperCreator;
                ITriangleMapCreator* lowerCreator;
                ITriangleMapCreator *bitmap;
                TriangleMapBalancedCreator *balanced;
                TriangleMapSAHCreator *sah;
                LowerAlgorithm lowerAlgorithm;

                bool propagateAabbs;
                float traversalCost;
                
                Resources::CUDA::CUDADataBlock<float4> *primMin;
                Resources::CUDA::CUDADataBlock<float4> *primMax;
                Resources::CUDA::CUDADataBlock<int> *primIndices;

                Resources::CUDA::CUDADataBlock<int> *leafIDs;
                
            public:
                TriangleMap(Scene::ISceneNode* scene);

                void Create();
                void Setup();

                LowerAlgorithm GetLowerAlgorithm() const { return lowerAlgorithm; }
                void SetLowerAlgorithm(const LowerAlgorithm l);
                float GetConstructionTime() const { return constructionTime; }
                void SplitEmptySpace(const bool s);
                bool IsSplittingEmptySpace() const;
                void SetSplitMethod(const SplitMethod s);
                SplitMethod GetSplitMethod();
                void SetPropagateBoundingBox(const bool p);
                bool GetPropagateBoundingBox() const { return propagateAabbs; }
                void SetTraversalCost(const float t);
                float GetTraversalCost() const { return traversalCost; }
                
                GeometryList* GetGeometry() const { return geom; }
                Scene::TriangleNode* GetNodes() const { return nodes; }
                Resources::CUDA::CUDADataBlock<int>* GetPrimitiveIndices() const { return primIndices; }

                void PrintTree();
                void PrintNode(int node, int offset = 0);
            };

        }
    }
}

#endif
