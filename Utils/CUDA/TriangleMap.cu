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

                // Initialized timer
                cutCreateTimer(&timerID);

                geom = new GeometryList(1);
                upperNodes = new TriangleUpperNode(1);

                scanConfig.algorithm = CUDPP_SCAN;
                scanConfig.op = CUDPP_ADD;
                scanConfig.datatype = CUDPP_INT;
                scanConfig.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;

                CUDPPResult res = cudppPlan(&scanHandle, scanConfig, triangles+1, 1, 0);
                if (CUDPP_SUCCESS != res)
                    throw Core::Exception("Error creating CUDPP scanPlan");

                scanInclConfig.algorithm = CUDPP_SCAN;
                scanInclConfig.op = CUDPP_ADD;
                scanInclConfig.datatype = CUDPP_INT;
                scanInclConfig.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;

                res = cudppPlan(&scanInclHandle, scanInclConfig, 1, 1, 0);
                if (CUDPP_SUCCESS != res)
                    throw Core::Exception("Error creating CUDPP inclusive scanPlan");
                

                tempAabbMin = new CUDADataBlock<1, point>(1);
                tempAabbMax = new CUDADataBlock<1, point>(1);
                segments = Segments(1);
                nodeSegments = new CUDADataBlock<1, int>(1);
            }

            void TriangleMap::Create(){

                Setup();
                
                CreateUpperNodes();
            }

            void TriangleMap::Setup(){
                int oldTris = triangles;
                
                geom->CollectGeometry(scene);
                triangles = geom->GetSize();

                logger.info << "Triangles " << triangles << logger.end;
                
                tempAabbMin->Resize(triangles);
                tempAabbMax->Resize(triangles);

                int approxSize = (2 * triangles / TriangleLowerNode::MAX_SIZE) - 1;
                upperNodes->Resize(approxSize);

                segments.Resize(triangles / Segments::SEGMENT_SIZE);

                if (oldTris < triangles){
                    //CUDPPResult res = cudppDestroyPlan(scanHandle);
                    //if (CUDPP_SUCCESS != res)
                    //throw Core::Exception("Error deleting CUDPP scanPlan");
                    
                    CUDPPResult res = cudppPlan(&scanHandle, scanConfig, triangles+1, 1, 0);
                    if (CUDPP_SUCCESS != res)
                        throw Core::Exception("Error creating CUDPP scanPlan");
                }
            }
            
        }
    }
}
