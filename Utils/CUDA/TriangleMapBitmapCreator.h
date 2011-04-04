// Triangle map bitmap creator/convertor interface for CUDA
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _TRIANGLE_MAP_BITMAP_CREATOR_H_
#define _TRIANGLE_MAP_BITMAP_CREATOR_H_

#include <Utils/CUDA/ITriangleMapCreator.h>

namespace OpenEngine {    
    namespace Utils {
        namespace CUDA {

            class TriangleMapBitmapCreator : public ITriangleMapCreator {
            protected:

            public:
                TriangleMapBitmapCreator();
                virtual ~TriangleMapBitmapCreator();

                virtual void Create(TriangleMap* map, 
                                    Resources::CUDA::CUDADataBlock<int>* upperLeafIDs);
            };
        }
    }
}

#endif
