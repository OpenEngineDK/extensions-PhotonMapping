// Triangle map class for CUDA
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Utils/CUDA/TriangleMap.h>
#include <Scene/ISceneNode.h>
#include <Scene/TriangleNode.h>
#include <Utils/CUDA/TriangleMapUpperCreator.h>
#include <Utils/CUDA/TriangleMapSAHCreator.h>
#include <Utils/CUDA/Convert.h>

using namespace OpenEngine::Scene;

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {

            TriangleMap::TriangleMap(ISceneNode* scene) 
                : scene(scene) {

                // Initialized timer
                cutCreateTimer(&timerID);

                geom = new GeometryList(1);
                nodes = new TriangleNode(1);
                
                primMin = new CUDADataBlock<1, float4>(1);
                primMax = new CUDADataBlock<1, float4>(1);
                primIndices = new CUDADataBlock<1, int>(1);

                leafIDs = new CUDADataBlock<1, int>(1);

                upperCreator = new TriangleMapUpperCreator();
                lowerCreator = new TriangleMapSAHCreator();
            }

            void TriangleMap::Create(){

                Setup();
                
                upperCreator->Create(this, NULL);

                lowerCreator->Create(this, leafIDs);

                /*
                for (int i = 0; i < nodes->GetSize(); ++i)
                    logger.info << nodes->ToString(i) << logger.end;
                */
            }

            void TriangleMap::Setup(){
                geom->CollectGeometry(scene);
            }
            
        }
    }
}
