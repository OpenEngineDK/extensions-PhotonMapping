// KD tree upper node for photons
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _CUDA_KD_BASE_NODE_H_
#define _CUDA_KD_BASE_NODE_H_

#include <Meta/CUDA.h>
#include <Utils/Cuda/Point.h>
#include <Resources/CUDA/CUDADataBlock.h>

#include <string>

using namespace OpenEngine::Resources::CUDA;

namespace OpenEngine {
    namespace Scene {
        
        class KDNode {
        public:
            static const char LEAF = 0;
            static const char X = 1;
            static const char Y = 2;
            static const char Z = 3;
            static const char PROXY = 1<<2;

            CUDADataBlock<1, char>* info; // 0 = LEAF,1 = X, 2 = Y, 3 = Z. 6 bits left for stuff
            CUDADataBlock<1, float>* splitPos;
            CUDADataBlock<1, point> *aabbMin, *aabbMax;
            CUDADataBlock<1, int2> *photonInfo; // [photonIndex, range/bitmap]

            // if it is a leaf node then both nodes
            // point to it's lower node.
            CUDADataBlock<1, int> *left, *right;
            
            int maxSize;
            int size;

        public:
            KDNode();
            KDNode(int i);
            virtual ~KDNode();

            virtual void Resize(int i);

            char* GetInfoData() { return info->GetDeviceData(); }
            float* GetSplitPositionData() { return splitPos->GetDeviceData(); }
            point* GetAabbMinData() { return aabbMin->GetDeviceData(); }
            point* GetAabbMaxData() { return aabbMax->GetDeviceData(); }
            int2* GetPrimitiveInfoData() { return photonInfo->GetDeviceData(); }
            int* GetLeftData() { return left->GetDeviceData(); }
            int* GetRightData() { return right->GetDeviceData(); }

            int GetSize() const { return size; }

            virtual std::string ToString(unsigned int i);
            
        };
        
    }
}

#endif
