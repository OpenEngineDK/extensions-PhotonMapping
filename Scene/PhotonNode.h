// Photon class for CUDA
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _PHOTON_NODE_H_
#define _PHOTON_NODE_H_

#include <Meta/CUDA.h>
#include <Utils/CUDA/Types.h>

#include <string>

namespace OpenEngine {
    namespace Resources {                
        class IDataBlock;
    }
    namespace Scene {                

        /**
         * Leaf and axis (plus even child) can be placed in one unsigned
         * int. Will probably be a slowdown but use less space.
         *
         * Place is-left or is-right node info in 'info' as a bit?
         */
        class PhotonNode {
        public:
            // Association list between indices used in the tree and actual
            // photons. This makes it impossible to coalesce memory
            // read/writes (for photons, some performance may be gained by
            // coalescing assoc lookups)
            //unsigned int *assoc;
            point* pos;
            unsigned int maxSize;
            unsigned int size;

        public:
            PhotonNode() 
                : pos(NULL), maxSize(0), size(0) {}
            PhotonNode(unsigned int size) 
                : maxSize(size), size(0) {
                cudaMalloc(&pos, maxSize * sizeof(point));
                CHECK_FOR_CUDA_ERROR();
            }

            std::string PositionToString(unsigned int begin, unsigned int range);

            void MapToDataBlocks(Resources::IDataBlock* vertices);
                    
        };
                
    }
}

#endif
