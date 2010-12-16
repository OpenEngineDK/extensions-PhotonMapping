// KD tree upper node for photons
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Scene/KDNode.h>
#include <Utils/CUDA/Utils.h>
#include <Utils/CUDA/Convert.h>
#include <Logging/Logger.h>

#include <sstream>

namespace OpenEngine {
    namespace Scene {
        
        KDNode::KDNode()
            : maxSize(0), size(0) {}

        KDNode::KDNode(int i)
            : maxSize(i), size(0) {
            
            info = new CUDADataBlock<1, char>(maxSize);
            splitPos = new CUDADataBlock<1, float>(maxSize);
            aabbMin = new CUDADataBlock<1, point>(maxSize);
            aabbMax = new CUDADataBlock<1, point>(maxSize);
            primitiveIndex = new CUDADataBlock<1, int>(maxSize);
            primitiveBitmap = new CUDADataBlock<1, bitmap>(maxSize);

            children = new CUDADataBlock<1, int2>(maxSize);

            CHECK_FOR_CUDA_ERROR();
        }

        KDNode::~KDNode(){
            logger.info << "Oh noes the KDNode is gone man" << logger.end;
            delete info;
            delete splitPos;
            delete aabbMin;
            delete aabbMax;
            delete primitiveIndex;
            delete primitiveBitmap;
            delete children;
        }

        void KDNode::Resize(int i){
            info->Extend(i);
            splitPos->Extend(i);
            aabbMin->Extend(i);
            aabbMax->Extend(i);
            primitiveIndex->Extend(i);
            primitiveBitmap->Extend(i);
            children->Extend(i);

            maxSize = i;
            size = i;
        }
        
        std::string KDNode::ToString(unsigned int i){
            bool isLeaf = false;
            std::ostringstream out;
                    
            out << "Upper node " << i << ":\n";
            char h_info;
            cudaMemcpy(&h_info, info->GetDeviceData() + i, sizeof(char), cudaMemcpyDeviceToHost);
            CHECK_FOR_CUDA_ERROR();
                    
            float h_pos;
            cudaMemcpy(&h_pos, splitPos->GetDeviceData() + i, sizeof(float), cudaMemcpyDeviceToHost);
            CHECK_FOR_CUDA_ERROR();
            switch(h_info){
            case X:
                out << "Splits along the X plane at pos " << h_pos << "\n";
                break;
            case Y:
                out << "Splits along the Y plane at pos " << h_pos << "\n";
                break;
            case Z:
                out << "Splits along the Z plane at pos " << h_pos << "\n";
                break;
            case LEAF:
                isLeaf = true;
                out << "Is a leaf\n";
                break;
            }

            if (isLeaf){
                int index;
                cudaMemcpy(&index, primitiveIndex->GetDeviceData() + i, sizeof(int), cudaMemcpyDeviceToHost);
                bitmap bmp;
                cudaMemcpy(&bmp, primitiveBitmap->GetDeviceData() + i, sizeof(bitmap), cudaMemcpyDeviceToHost);
                out << "Index " << index << " and range " << bmp << " " << BitmapToString(bmp) << "\n";
                CHECK_FOR_CUDA_ERROR();
            }
            
            //if (!isLeaf){
                point h_aabbmin, h_aabbmax;
                cudaMemcpy(&h_aabbmin, aabbMin->GetDeviceData() + i, sizeof(point), cudaMemcpyDeviceToHost);
                cudaMemcpy(&h_aabbmax, aabbMax->GetDeviceData() + i, sizeof(point), cudaMemcpyDeviceToHost);
                CHECK_FOR_CUDA_ERROR();
                out << "Axis aligned bounding box: " << Utils::CUDA::Convert::ToString(h_aabbmin);
                out << " -> " << Utils::CUDA::Convert::ToString(h_aabbmax) << "\n";
                //}


            int2 h_children;
            cudaMemcpy(&h_children, children->GetDeviceData() + i, sizeof(int2), cudaMemcpyDeviceToHost);
            CHECK_FOR_CUDA_ERROR();
            if (!isLeaf){
                out << "Has children " << h_children.x << " and " << h_children.y << "\n";
            }
                    
            return out.str();
        }

    }
}
