// Compute Bounding boxes for all upper nodes
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Meta/CUDA.h>
#include <Utils/CUDA/Point.h>
#include <Utils/CUDA/Kernels/PhotonMapDeviceVars.h>

namespace OpenEngine {
namespace Utils {
namespace CUDA {
namespace Kernels {

    __global__ void ConstantTimeBoundingBox(int2 *photonInfo, // [index, range]
                                            point *aabbMin, point *aabbMax,
                                            float4 *xSorted, float4 *ySorted, float4 *zSorted){

        int id = blockDim.x * blockIdx.x + threadIdx.x;
        int stepSize = gridDim.x * blockDim.x;

        while (id < d_activeNodeRange){
            
            int2 info = photonInfo[id];
            int photonEnd = info.x + info.y - 1;
            
            aabbMin[id] = make_point(xSorted[info.x].x,
                                     ySorted[info.x].y,
                                     zSorted[info.x].z);

            aabbMax[id] = make_point(xSorted[photonEnd].x,
                                     ySorted[photonEnd].y,
                                     zSorted[photonEnd].z);

            id += stepSize;
        }
    }
    
    /**
     * Requires none of the active nodes are leafs (should be split
     * away).
     */
    __global__ void SetUpperNodeSplitInfo(point *aabbMin, point *aabbMax,
                                          float *splitPos, char *info){
        int id = blockDim.x * blockIdx.x + threadIdx.x;
        int stepSize = gridDim.x * blockDim.x;
        
        while (id < d_activeNodeRange){
            
            point bbSize = aabbMax[id] - aabbMin[id];
            point median = bbSize * 0.5 + aabbMin[id];

            // Calculate splitting plane
            bool yAboveX = bbSize.x < bbSize.y;
            float max = yAboveX ? bbSize.y : bbSize.x;
            float split = yAboveX ? median.y : median.x;
            char axis = yAboveX ? KDNode::Y : KDNode::X;
            bool zHigher = max < bbSize.z;
            splitPos[id] = zHigher ? median.z : split;
            info[id] = zHigher ? KDNode::Z : axis;
            
            id += stepSize;
        }
    }

}
}
}
}
