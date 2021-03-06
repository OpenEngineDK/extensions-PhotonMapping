// Reduces photon position to a bounding box
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _PHOTON_MAP_DEVICE_VARS_H_
#define _PHOTON_MAP_DEVICE_VARS_H_

#include <Meta/CUDA.h>
#include <Utils/CUDA/Point.h>

namespace OpenEngine {
namespace Utils {
namespace CUDA {
namespace Kernels {

    __constant__ int d_triangles;

    __constant__ int d_segments;
    __constant__ int d_activeNodeIndex;
    __constant__ int d_activeNodeRange;
    __constant__ int d_childIndex;

    __constant__ int d_leafNodes;

    __device__ bool d_createdLeafs;
    __device__ int d_leafsCreated;

    __device__ bool d_createdEmptySplits;

    __constant__ float d_emptySpaceThreshold;
}
}
}
}

#endif
