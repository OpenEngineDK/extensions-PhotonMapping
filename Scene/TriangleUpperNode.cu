// KD tree upper node for triangles
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Scene/TriangleUpperNode.h>

#include <sstream>

namespace OpenEngine {
    namespace Scene {
        
        TriangleUpperNode::TriangleUpperNode()
            : KDNode() {}

        TriangleUpperNode::TriangleUpperNode(int size)
            : KDNode(size) {
            parent = new CUDADataBlock<1, int>(maxSize);
        }

        void TriangleUpperNode::Resize(int i){
            KDNode::Resize(i);
            parent->Resize(i);
        }

        std::string TriangleUpperNode::ToString(unsigned int i){
            std::ostringstream out;

            out << KDNode::ToString(i);

            int h_parent;
            cudaMemcpy(&h_parent, parent->GetDeviceData() + i, sizeof(int), cudaMemcpyDeviceToHost);
            out << "Has parent " << h_parent << "\n";

            return out.str();
        }

    }
}
