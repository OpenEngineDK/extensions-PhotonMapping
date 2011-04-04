// KD tree node for triangles
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _CUDA_TRIANGLE_NODE_H_
#define _CUDA_TRIANGLE_NODE_H_

#include <Scene/KDNode.h>

namespace OpenEngine {
    namespace Scene {

        class TriangleNode : public KDNode {
        public:
            CUDADataBlock<float> *surfaceArea;
            CUDADataBlock<int> *parent;
            
        public:
            TriangleNode();
            TriangleNode(int size);

            void Resize(int i);
            void Extend(int i);

            float* GetSurfaceAreaData() const { return surfaceArea->GetDeviceData(); }
            int* GetParentData() const { return parent->GetDeviceData(); }
            
            virtual std::string ToString(unsigned int i);
        };

    }
}

#endif
