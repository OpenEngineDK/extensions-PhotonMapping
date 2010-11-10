// Triangle map median split creator
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _TRIANGLE_MAP_MEDIAN_SPLIT_H_
#define _TRIANGLE_MAP_MEDIAN_SPLIT_H_

#include <Utils/CUDA/ITriangleMapCreator.h>

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {

            class TriangleMapMedianSplit : public ITriangleMapCreator {
            private:
                

            public:
                TriangleMapMedianSplit();
                virtual ~TriangleMapMedianSplit();

                void Create(Scene::TriangleNode* nodes, 
                            int activeIndex, int activeRange,
                            ITriangleMapCreator* upper);
            };

        }
    }
}

#endif
