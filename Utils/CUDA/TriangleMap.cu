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
#include <Utils/CUDA/TriangleMapBalancedCreator.h>
#include <Utils/CUDA/TriangleMapBitmapCreator.h>
#include <Utils/CUDA/Utils.h>
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
                //lowerCreator = new TriangleMapBalancedCreator();
                //lowerCreator = new TriangleMapSAHCreator();
                lowerCreator = new TriangleMapBitmapCreator();
            }

            void TriangleMap::Create(){
                Setup();

                //logger.info << "Start primitives: " << geom->GetSize() << logger.end;

                START_TIMER(timerID);
                upperCreator->Create(this, NULL);

                lowerCreator->Create(this, leafIDs);

                //PRINT_TIMER(timerID, "Total tree creation");
                cudaThreadSynchronize();
                cutStopTimer(timerID);
                constructionTime = cutGetTimerValue(timerID);


                //logger.info << "End primitives: " << primIndices->GetSize() << logger.end;
                //logger.info << "Tree nodes: " << nodes->GetSize() << logger.end;

                /*
                for (int i = 0; i < nodes->GetSize(); ++i)
                    logger.info << nodes->ToString(i) << logger.end;
                */
                //exit(0);
            }

            void TriangleMap::Setup(){
                geom->CollectGeometry(scene);
            }
            
            void TriangleMap::PrintTree(){
                PrintNode(0);
            }
            
            void TriangleMap::PrintNode(int node, int offset){
                for (int i = 0; i < offset; ++i)
                    logger.info << " ";

                char info = FetchGlobalData(nodes->GetInfoData(), node);
                if (info == KDNode::LEAF){
                    int index = FetchGlobalData(nodes->GetPrimitiveIndexData(), node);
                    KDNode::bitmap bmp = FetchGlobalData(nodes->GetPrimitiveBitmapData(), node);
                    logger.info << "leaf " << node << " with primitives ";
                    
                    while (bmp){
                        int i = firstBitSet(bmp) - 1;
                        bmp -= KDNode::bitmap(1)<<i;
                      
                        int prim = FetchGlobalData(primIndices->GetDeviceData(), index + i);
                        logger.info << prim;
                        if (bmp) logger.info << ", ";
                    }
                    logger.info << logger.end;
                }else{
                    logger.info << "node " << node << " splits along ";
                    switch(info & 3){
                    case KDNode::X:
                        logger.info << "x:";
                        break;
                    case KDNode::Y:
                        logger.info << "y:";
                        break;
                    case KDNode::Z:
                        logger.info << "z:";
                        break;
                    }
                    logger.info << FetchGlobalData(nodes->GetSplitPositionData(), node) << logger.end;
                    int2 children = FetchGlobalData(nodes->GetChildrenData(), node);
                    PrintNode(children.x, offset+1);
                    PrintNode(children.y, offset+1);
                }
            }
            
        }
    }
}
