// Triangle map bitmap creator/convertor interface for CUDA
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Utils/CUDA/TriangleMapBitmapCreator.h>

#include <Scene/TriangleNode.h>
#include <Utils/CUDA/TriangleMap.h>
#include <Utils/CUDA/Utils.h>
#include <Utils/CUDA/Convert.h>

using namespace OpenEngine::Resources::CUDA;
using namespace OpenEngine::Scene;

namespace OpenEngine {    
    namespace Utils {
        namespace CUDA {

            TriangleMapBitmapCreator::TriangleMapBitmapCreator(){

            }
            
            TriangleMapBitmapCreator::~TriangleMapBitmapCreator(){

            }

            __global__ void PreprocessLeafNodes(int *upperLeafIDs,
                                                char *axis,
                                                KDNode::bitmap *primBitmap,
                                                int activeRange){
                
                const int id = blockDim.x * blockIdx.x + threadIdx.x;
                
                if (id < activeRange){
                    int leafID = upperLeafIDs[id];
                    
                    axis[leafID] = KDNode::LEAF;

                    KDNode::bitmap bmp = primBitmap[leafID];
                    bmp = (KDNode::bitmap(1)<<bmp)-1;
                    
                    primBitmap[leafID] = bmp;
                }
            }

            void TriangleMapBitmapCreator::Create(TriangleMap* map, 
                                                  CUDADataBlock<int>* upperLeafIDs){

                //logger.info << "=== Convert " << upperLeafIDs->GetSize() << " bitmaps ===" << logger.end;

                SetPropagateBoundingBox(map->GetPropagateBoundingBox());

                KernelConf conf = KernelConf1D(upperLeafIDs->GetSize());
                PreprocessLeafNodes<<<conf.blocks, conf.threads>>>
                    (upperLeafIDs->GetDeviceData(), 
                     map->GetNodes()->GetInfoData(),
                     map->GetNodes()->GetPrimitiveBitmapData(),
                     upperLeafIDs->GetSize());
                CHECK_FOR_CUDA_ERROR();
            }

        }
    }
}
