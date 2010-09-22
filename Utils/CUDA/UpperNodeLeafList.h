// KD tree upper node leaf list for photons
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _CUDA_UPPER_NODE_LEAFS_LIST_H_
#define _CUDA_UPPER_NODE_LEAFS_LIST_H_

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {

            class UpperNodeLeafList {
            public:
                int maxSize, size;
                int *leafIDs;
                
            public:
                UpperNodeLeafList()
                    : maxSize(0), size(0), leafIDs(NULL) {}
                UpperNodeLeafList(int size);

                void Resize(int size);
            };

        }
    }
}

#endif
