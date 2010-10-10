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
                : scene(scene), triangles(1) {
                geom = new GeometryList(1);
                upperNodes = new TriangleUpperNode(1);

                scanConfig.algorithm = CUDPP_SCAN;
                scanConfig.op = CUDPP_ADD;
                scanConfig.datatype = CUDPP_INT;
                scanConfig.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;

                CUDPPResult res = cudppPlan(&scanHandle, scanConfig, triangles+1, 1, 0);
                if (CUDPP_SUCCESS != res)
                    throw Core::Exception("Error creating CUDPP scanPlan");

                tempAabbMin = new CUDADataBlock<1, point>(1);
                tempAabbMax = new CUDADataBlock<1, point>(1);
                segments = Segments(1);
                nodeSegments = new CUDADataBlock<1, int>(1);
            }

            void TriangleMap::Create(){

                Setup();

                int activeIndex = 0, activeRange = 1;
                int newActiveIndex, childrenCreated;
                
                while (activeRange > 0){
                    ProcessUpperNodes(activeIndex, activeRange, 
                                      newActiveIndex, childrenCreated);

                    // @TODO Isn't active index = upperNodes.size - childrenCreated?
                    activeIndex = newActiveIndex;
                    activeRange = childrenCreated;
                }
            }

            void TriangleMap::Setup(){
                int oldTris = triangles;
                
                geom->CollectGeometry(scene);
                int tris = geom->GetSize();

                logger.info << "Triangles " << tris << logger.end;
                
                tempAabbMin->Resize(tris);
                tempAabbMax->Resize(tris);

                int approxSize = (2 * tris / TriangleLowerNode::MAX_SIZE) - 1;
                upperNodes->Resize(approxSize);

                segments.Resize(tris / Segments::SEGMENT_SIZE);

                if (oldTris < tris){
                    //CUDPPResult res = cudppDestroyPlan(scanHandle);
                    //if (CUDPP_SUCCESS != res)
                    //throw Core::Exception("Error deleting CUDPP scanPlan");
                    
                    CUDPPResult res = cudppPlan(&scanHandle, scanConfig, tris+1, 1, 0);
                    if (CUDPP_SUCCESS != res)
                        throw Core::Exception("Error creating CUDPP scanPlan");
                }
            }
            
        }
    }
}
