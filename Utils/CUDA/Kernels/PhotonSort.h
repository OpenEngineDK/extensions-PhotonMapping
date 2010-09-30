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
                            int* xIndices, int* yIndices, int* zIndices,
                            float* xKeys, float* yKeys, float* zKeys, 
                            int size){
        
        const int id = blockDim.x * blockIdx.x + threadIdx.x;

        if (id < size){

            point pos = positions[id];
            xIndices[id] = yIndices[id] = zIndices[id] = id;
            xKeys[id] = pos.x;
            yKeys[id] = pos.y;
            zKeys[id] = pos.z;
        }
    }

    __global__ void ScatterPhotons(point *positions,
                                   int* xIndices, int* yIndices, int* zIndices,
                                   float4 *xSorted, float4 *ySorted, float4 *zSorted,
                                   int size){
        
        const int id = blockDim.x * blockIdx.x + threadIdx.x;

        if (id < size){

            int index = xIndices[id];
            point pos = positions[index];
            xSorted[id] = make_float4(pos.x, pos.y, pos.z, index);

            index = yIndices[id];
            pos = positions[index];
            ySorted[id] = make_float4(pos.x, pos.y, pos.z, index);
            
            index = zIndices[id];
            pos = positions[index];
            zSorted[id] = make_float4(pos.x, pos.y, pos.z, index);
        }
    }

    __global__ void ScatterPhotons(int *indices,
                                    point *position,
                                    point *newPosition,
                                    int size){
        
        const int id = blockDim.x * blockIdx.x + threadIdx.x;

        if (id < size){
            const int oldId = indices[id];
            newPosition[id] = position[oldId];
        }
    }

}
}
}
}
