// Kernels for segmenting triangle upper nodes
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Meta/CUDA.h>
#include <Scene/TriangleLowerNode.h>
#include <Utils/CUDA/Kernels/PhotonMapDeviceVars.h>

namespace OpenEngine {
namespace Utils {
namespace CUDA {
namespace Kernels {

    __global__ void NodeSegments(int2* primitiveInfo,
                                 int* nodeSegments){
        const int id = blockDim.x * blockIdx.x + threadIdx.x;
        
        if (id < d_activeNodeRange){
            nodeSegments[id] = (primitiveInfo[id].x+1) / Scene::TriangleLowerNode::MAX_SIZE;
        }
    }

}
}
}
}
