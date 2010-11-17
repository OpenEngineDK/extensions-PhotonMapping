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
            photonInfo = new CUDADataBlock<1, int2>(maxSize);

            left = new CUDADataBlock<1, int>(maxSize);
            right = new CUDADataBlock<1, int>(maxSize);

            CHECK_FOR_CUDA_ERROR();
        }

        KDNode::~KDNode(){
            logger.info << "Oh noes the KDNode is gone man" << logger.end;
            delete info;
            delete splitPos;
            delete aabbMin;
            delete aabbMax;
            delete photonInfo;
            delete left;
            delete right;
        }

        void KDNode::Resize(int i){
            info->Resize(i);
            splitPos->Resize(i);
            aabbMin->Resize(i);
            aabbMax->Resize(i);
            photonInfo->Resize(i);
            left->Resize(i);
            right->Resize(i);

            maxSize = i;
            size = min(i, size);
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
                int2 info;
                cudaMemcpy(&info, photonInfo->GetDeviceData() + i, sizeof(int2), cudaMemcpyDeviceToHost);
                out << "Index " << info.x << " and range " << info.y << " " << BitmapToString(info.y) << "\n";
                CHECK_FOR_CUDA_ERROR();
            }
            
            if (!isLeaf){
                point h_aabbmin, h_aabbmax;
                cudaMemcpy(&h_aabbmin, aabbMin->GetDeviceData() + i, sizeof(point), cudaMemcpyDeviceToHost);
                cudaMemcpy(&h_aabbmax, aabbMax->GetDeviceData() + i, sizeof(point), cudaMemcpyDeviceToHost);
                CHECK_FOR_CUDA_ERROR();
                out << "Axis aligned bounding box: " << Utils::CUDA::Convert::ToString(h_aabbmin);
                out << " -> " << Utils::CUDA::Convert::ToString(h_aabbmax) << "\n";
            }


            int h_left, h_right;
            cudaMemcpy(&h_left, left->GetDeviceData() + i, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_right, right->GetDeviceData() + i, sizeof(int), cudaMemcpyDeviceToHost);
            CHECK_FOR_CUDA_ERROR();
            if (!isLeaf){
                out << "Has children " << h_left << " and " << h_right << "\n";
            }
                    
            return out.str();
        }

    }
}
