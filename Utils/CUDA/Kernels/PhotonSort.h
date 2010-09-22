// Kernels for sorting upper node children
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Meta/CUDA.h>

namespace OpenEngine {
namespace Utils {
namespace CUDA {
namespace Kernels {

    __global__ void Indices(point *positions,
                            float* xIndices, float* yIndices, float* zIndices,
                            float* xKeys, float* yKeys, float* zKeys, 
                            int size){
        
        int id = blockDim.x * blockIdx.x + threadIdx.x;
        //int id = globalIdx2D(threadIdx, blockIdx, blockDim, gridDim);
        int stepSize = gridDim.x * blockDim.x;

        while (id < size){

            point pos = positions[id];
            xIndices[id] = yIndices[id] = zIndices[id] = id;
            xKeys[id] = pos.x;
            yKeys[id] = pos.y;
            zKeys[id] = pos.z;
            
            id += stepSize;
        }
    }
    __global__ void ScatterPhotons(point *positions,
                                   float* xIndices, float* yIndices, float* zIndices,
                                   float4 *xSorted, float4 *ySorted, float4 *zSorted,
                                   int size){
        
        int id = blockDim.x * blockIdx.x + threadIdx.x;
        int stepSize = gridDim.x * blockDim.x;

        while (id < size){

            float index = xIndices[id];
            point pos = positions[(int)index];
            xSorted[id] = make_float4(pos.x, pos.y, pos.z, index);

            index = yIndices[id];
            pos = positions[(int)index];
            ySorted[id] = make_float4(pos.x, pos.y, pos.z, index);
            
            index = zIndices[id];
            pos = positions[(int)index];
            zSorted[id] = make_float4(pos.x, pos.y, pos.z, index);

            id += stepSize;
        }
    }

}
}
}
}
