// KD tree node for triangles
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Scene/TriangleNode.h>

#include <sstream>

namespace OpenEngine {
    namespace Scene {
        
        TriangleNode::TriangleNode()
            : KDNode() {}

        TriangleNode::TriangleNode(int size)
            : KDNode(size) {
            surfaceArea = new CUDADataBlock<float>(maxSize);
            parent = new CUDADataBlock<int>(maxSize);
        }

        void TriangleNode::Resize(int i){
            KDNode::Resize(i);
            surfaceArea->Resize(i);
            parent->Resize(i);
        }

        void TriangleNode::Extend(int i){
            if (this->maxSize < i) Resize(i);
        }

        std::string TriangleNode::ToString(unsigned int i){
            std::ostringstream out;

            out << KDNode::ToString(i);

            int h_parent;
            cudaMemcpy(&h_parent, parent->GetDeviceData() + i, sizeof(int), cudaMemcpyDeviceToHost);
            out << "Has parent " << h_parent << "\n";

            float h_area;
            cudaMemcpy(&h_area, surfaceArea->GetDeviceData() + i, sizeof(float), cudaMemcpyDeviceToHost);
            out << "Has surface area " << h_area << "\n";

            return out.str();
        }

    }
}
