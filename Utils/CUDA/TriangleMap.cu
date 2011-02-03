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
#include <Utils/CUDA/LoggerExtensions.h>

using namespace OpenEngine::Scene;

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {

            TriangleMap::TriangleMap(ISceneNode* scene)
                : scene(scene){

                // Initialized timer
                cutCreateTimer(&timerID);

                geom = new GeometryList(1);
                nodes = new TriangleNode(1);
                
                primMin = new CUDADataBlock<1, float4>(1);
                primMax = new CUDADataBlock<1, float4>(1);
                primIndices = new CUDADataBlock<1, int>(1);

                leafIDs = new CUDADataBlock<1, int>(1);

                upperCreator = new TriangleMapUpperCreator();
                balanced = new TriangleMapBalancedCreator();
                sah = new TriangleMapSAHCreator();
                bitmap = new TriangleMapBitmapCreator();
                SetLowerAlgorithm(BITMAP);
                SetPropagateBoundingBox(true);
            }

            void TriangleMap::Create(){
                Setup();

                START_TIMER(timerID);
                upperCreator->Create(this, NULL);

                lowerCreator->Create(this, leafIDs);

                cudaThreadSynchronize();
                cutStopTimer(timerID);
                constructionTime = cutGetTimerValue(timerID);
            }

            void TriangleMap::Setup(){
                geom->CollectGeometry(scene);
            }

            void TriangleMap::SetLowerAlgorithm(const LowerAlgorithm l){
                lowerAlgorithm = l;
                switch(lowerAlgorithm){
                case BITMAP:
                    logger.info << "Switching to bitmap converter lower creator" << logger.end;
                    lowerCreator = bitmap; break;
                case BALANCED:
                    logger.info << "Switching to balanced lower creator" << logger.end;
                    lowerCreator = balanced; break;
                case SAH:
                    logger.info << "Switching to SAH lower creator" << logger.end;
                    lowerCreator = sah; break;
                }
            }

            void TriangleMap::SplitEmptySpace(const bool s) { 
                upperCreator->SplitEmptySpace(s); 
            }
            bool TriangleMap::IsSplittingEmptySpace() const { 
                return upperCreator->IsSplittingEmptySpace(); 
            }

            void TriangleMap::SetSplitMethod(const SplitMethod s) { upperCreator->SetSplitMethod(s); }

            TriangleMap::SplitMethod TriangleMap::GetSplitMethod() { 
                return upperCreator->GetSplitMethod(); 
            }
            
            void TriangleMap::SetPropagateBoundingBox(const bool p) {
                propagateAabbs = p; 
                upperCreator->SetPropagateBoundingBox(p);
                //bitmap->SetPropagateBoundingBox(p);
                //balanced->SetPropagateBoundingBox(p);
                //sah->SetPropagateBoundingBox(p);
            }

            void TriangleMap::PrintTree(){
                PrintNode(0);

                logger.info << "Start primitives: " << geom->GetSize() << logger.end;
                logger.info << "End primitives: " << primIndices->GetSize() << logger.end;
                logger.info << "Tree nodes: " << nodes->GetSize() << logger.end;
            }
            
            void TriangleMap::PrintNode(int node, int offset){
                for (int i = 0; i < offset; ++i)
                    logger.info << " ";

                char info = FetchGlobalData(nodes->GetInfoData(), node);
                if (info == KDNode::LEAF){
                    int index = FetchGlobalData(nodes->GetPrimitiveIndexData(), node);
                    KDNode::bitmap bmp = FetchGlobalData(nodes->GetPrimitiveBitmapData(), node);
                    logger.info << "leaf " << node << " with " << bitcount(bmp) << " primitives ";

                    // logger.info << "leaf " << node << " with primitives ";
                    // while (bmp){
                    //     int i = firstBitSet(bmp) - 1;
                    //     bmp -= KDNode::bitmap(1)<<i;
                    //     int prim = FetchGlobalData(primIndices->GetDeviceData(), index + i);
                    //     logger.info << prim;
                    //     if (bmp) logger.info << ", ";
                    // }

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
                    logger.info << FetchGlobalData(nodes->GetSplitPositionData(), node);
                    
                    // float3 nodeMin = make_float3(FetchGlobalData(nodes->GetAabbMinData(), node));
                    // float3 nodeMax = make_float3(FetchGlobalData(nodes->GetAabbMaxData(), node));
                    // logger.info << " aabb: " << nodeMin << " -> " << nodeMax;

                    logger.info << logger.end;
                    int2 children = FetchGlobalData(nodes->GetChildrenData(), node);
                    PrintNode(children.x, offset+1);
                    PrintNode(children.y, offset+1);
                }
            }
            
        }
    }
}
