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

using namespace OpenEngine::Scene;

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {

            TriangleMap::TriangleMap(ISceneNode* scene) 
                : scene(scene) {
                geom = new GeometryList(1);
                tempAabbMin = new CUDADataBlock<1, point>(1);
                tempAabbMax = new CUDADataBlock<1, point>(1);
                segments = Segments(1);
                upperNodes = new TriangleUpperNode(1);
            }

            void TriangleMap::Create(){
                geom->CollectGeometry(scene);
                int tris = geom->GetSize();
                
                tempAabbMin->Resize(tris);
                tempAabbMax->Resize(tris);

                int approxSize = (2 * tris / TriangleLowerNode::MAX_SIZE) - 1;
                upperNodes->Resize(approxSize);

                segments.Resize(tris / Segments::SEGMENT_SIZE);

                int activeIndex = 0, activeRange = 1;
                int leafsCreated, childrenCreated;
                
                while (activeRange > 0){
                    ProcessUpperNodes(activeIndex, activeRange, 
                                      leafsCreated, childrenCreated);

                    activeIndex += activeRange + leafsCreated;
                    activeRange = childrenCreated;
                }
            }

        }
    }
}
