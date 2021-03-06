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
#include <Utils/CUDA/Point.h>
#include <Resources/CUDA/CUDADataBlock.h>

#include <string>

using namespace OpenEngine::Resources::CUDA;

#define KDNODE_MAX_LOWER_SIZE 32

namespace OpenEngine {
    namespace Scene {

        class KDNode {
        public: 

            static const int MAX_LOWER_SIZE = KDNODE_MAX_LOWER_SIZE;

#if KDNODE_MAX_LOWER_SIZE <= 32
            typedef int bitmap;
            typedef int2 bitmap2;
            typedef int4 bitmap4;
#elif KDNODE_MAX_LOWER_SIZE <= 64
            typedef long long int bitmap;
            typedef longlong2 bitmap2;
            typedef longlong4 bitmap4;
#endif

            typedef bitmap amount;

            static __host__ __device__ bitmap2 make_bitmap2(bitmap x, bitmap y){
                bitmap2 bmp2;
                bmp2.x = x;
                bmp2.y = y;
                return bmp2;
            }

            static __host__ __device__ bitmap4 make_bitmap4(bitmap x, bitmap y, bitmap z, bitmap w){
                bitmap4 bmp;
                bmp.x = x;
                bmp.y = y;
                bmp.z = z;
                bmp.w = w;
                return bmp;
            }

            static const char LEAF = 0;
            static const char X = 1;
            static const char Y = 2;
            static const char Z = 3;

        protected:
            CUDADataBlock<char>* info; // 0 = LEAF,1 = X, 2 = Y, 3 = Z. 6 bits left for stuff
            CUDADataBlock<float>* splitPos;
            CUDADataBlock<point> *aabbMin, *aabbMax;
            CUDADataBlock<int> *primitiveIndex;
            CUDADataBlock<bitmap> *primitiveBitmap;

            // if it is a leaf node then both nodes
            // point to it's lower node.
            CUDADataBlock<int2> *children;
            
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
            int* GetPrimitiveIndexData() { return primitiveIndex->GetDeviceData(); }
            amount* GetPrimitiveAmountData() { return primitiveBitmap->GetDeviceData(); }
            bitmap* GetPrimitiveBitmapData() { return primitiveBitmap->GetDeviceData(); }
            int2* GetChildrenData() { return children->GetDeviceData(); }

            int GetSize() const { return size; }

            virtual std::string ToString(unsigned int i);
            
        };
        
    }
}

#endif
