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
                : scene(scene), triangles(1), emptySpaceThreshold(0.25f) {

                // Initialized timer
                cutCreateTimer(&timerID);

                geom = new GeometryList(1);
                nodes = new TriangleNode(1);

                scanConfig.algorithm = CUDPP_SCAN;
                scanConfig.op = CUDPP_ADD;
                scanConfig.datatype = CUDPP_INT;
                scanConfig.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
                scanSize = triangles+1;
                
                CUDPPResult res = cudppPlan(&scanHandle, scanConfig, scanSize, 1, 0);
                if (CUDPP_SUCCESS != res)
                    throw Core::Exception("Error creating CUDPP scanPlan");

                scanInclConfig.algorithm = CUDPP_SCAN;
                scanInclConfig.op = CUDPP_ADD;
                scanInclConfig.datatype = CUDPP_INT;
                scanInclConfig.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
                scanInclSize = 1;

                res = cudppPlan(&scanInclHandle, scanInclConfig, scanInclSize, 1, 0);
                if (CUDPP_SUCCESS != res)
                    throw Core::Exception("Error creating CUDPP inclusive scanPlan");
                
                aabbMin = new CUDADataBlock<1, float4>(1);
                aabbMax = new CUDADataBlock<1, float4>(1);
                tempAabbMin = new CUDADataBlock<1, float4>(1);
                tempAabbMax = new CUDADataBlock<1, float4>(1);
                resultMin = new CUDADataBlock<1, float4>(1);
                resultMax = new CUDADataBlock<1, float4>(1);

                segments = Segments(1);
                nodeSegments = new CUDADataBlock<1, int>(1);

                splitSide = new CUDADataBlock<1, int>(1);
                splitAddr = new CUDADataBlock<1, int>(1);
                leafSide = new CUDADataBlock<1, int>(1);
                leafAddr = new CUDADataBlock<1, int>(1);
                emptySpaceSplits = new CUDADataBlock<1, int>(1);
                emptySpaceAddrs = new CUDADataBlock<1, int>(1);
                childSize = new CUDADataBlock<1, int2>(1);

                upperNodeLeafList = new CUDADataBlock<1, int>(1);

                splitTriangleSet =  new CUDADataBlock<1, int4>(1);

                childAreas = new CUDADataBlock<1, float2>(1);
                childSets = childSize;
            }

            void TriangleMap::Create(){

                Setup();
                
                CreateUpperNodes();

                CreateLowerNodes();
            }

            void TriangleMap::Setup(){
                int oldTris = triangles;
                
                geom->CollectGeometry(scene);
                triangles = geom->GetSize();

                logger.info << "Triangles " << triangles << logger.end;
                
                aabbMin->Resize(triangles);
                aabbMax->Resize(triangles);
                tempAabbMin->Resize(triangles);
                tempAabbMax->Resize(triangles);
                resultMin->Resize(triangles);
                resultMax->Resize(triangles);

                int approxSize = (2 * triangles / TriangleNode::MAX_LOWER_SIZE) - 1;
                nodes->Resize(approxSize);

                segments.Resize(triangles / Segments::SEGMENT_SIZE);

                if (oldTris < triangles){
                    //CUDPPResult res = cudppDestroyPlan(scanHandle);
                    //if (CUDPP_SUCCESS != res)
                    //throw Core::Exception("Error deleting CUDPP scanPlan");
                    
                    scanSize = triangles+1;
                    CUDPPResult res = cudppPlan(&scanHandle, scanConfig, scanSize, 1, 0);
                    if (CUDPP_SUCCESS != res)
                        throw Core::Exception("Error creating CUDPP scanPlan");
                }
            }
            
        }
    }
}
