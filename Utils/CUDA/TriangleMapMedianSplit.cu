// Triangle map median split creator
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Utils/CUDA/TriangleMapMedianSplit.h>

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {
            
            TriangleMapMedianSplit::TriangleMapMedianSplit()
                : ITriangleMapCreator() {
                
            }

            TriangleMapMedianSplit::~TriangleMapMedianSplit(){

            }

            void TriangleMapMedianSplit::Create(Scene::TriangleNode* nodes,
                                                int activeIndex, int activeRange,
                                                ITriangleMapCreator* upper){

            }


        }
    }
}
