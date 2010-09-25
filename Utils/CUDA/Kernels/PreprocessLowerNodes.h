// Preprocess Lower Nodes.
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Meta/CUDA.h>
#include <Utils/CUDA/SharedMemory.h>
#include <Utils/CUDA/Utils.h>
#include <Scene/KDNode.h>
#include <Scene/PhotonLowerNode.h>
#include <Utils/CUDA/SharedMemory.h>

using namespace OpenEngine::Scene;

namespace OpenEngine {
namespace Utils {
namespace CUDA {
namespace Kernels {

    
    __global__ void CreateLowerNodes(int *upperLeafIDs,
                                     int2 *upperPhotonInfo,
                                     char *lowerInfo,
                                     int2 *lowerPhotonInfo,
                                     int lowerNodes){

        const int id = blockDim.x * blockIdx.x + threadIdx.x;
                
        if (id < lowerNodes){
            int upperID = upperLeafIDs[id];
            lowerInfo[id] = KDNode::LEAF;
            int2 photonInfo = upperPhotonInfo[upperID];
            // Mark the n lowest bits
            photonInfo.y = (1<<photonInfo.y)-1;
            lowerPhotonInfo[id] = photonInfo;
        }
    }

    __global__ void CreateSplittingPlanes(int2 *splitTriangleSetX,
                                          int2 *splitTriangleSetY,
                                          int2 *splitTriangleSetZ,
                                          int2 *lowerPhotonInfo,
                                          float4 *positions,
                                          int lowerNodes){

        const int id = blockDim.x * blockIdx.x + threadIdx.x;
        const int nodeID = id / PhotonLowerNode::MAX_SIZE;
        const int photonID = id % PhotonLowerNode::MAX_SIZE;
        const int2 photonInfo = lowerPhotonInfo[nodeID];
        const int photons = bitcount(photonInfo.y);
        
        // Copy photons to shared memory
        __shared__ float4 photonPos[512];
        photonPos[threadIdx.x] = photonID < photons ? positions[photonInfo.x + photonID] : make_float4(0.0f);
        /*
        if (photonID < photons)
            photonPos[threadIdx.x] = positions[photonInfo.x + photonID];
        else
            photonPos[threadIdx.x] = make_float4(0.0f);
        */
        __syncthreads();

        /*

        if (nodeID < lowerNodes){
            // Since we're using points, split
            int splitX = 0, splitY = 0, splitZ = 0;

            const float4 splitPlane = photonPos[threadIdx.x];

            // @OPT unroll MAX_SIZE photons instead? There is enough
            // positions in shared mem and the extra bits in the mask
            // doens't matter.
            for (int i = 0; i < photons; ++i){
                int index = nodeID * PhotonLowerNode::MAX_SIZE + i;
                splitX += photonPos[index].x < splitPlane.x ? 1<<i : 0;
                splitY += photonPos[index].y < splitPlane.y ? 1<<i : 0;
                splitZ += photonPos[index].z < splitPlane.z ? 1<<i : 0;
            }

            if (photonID < photons){
                splitTriangleSetX[photonInfo.x + photonID] = make_int2(splitX, ~splitX);
                splitTriangleSetY[photonInfo.x + photonID] = make_int2(splitY, ~splitY);
                splitTriangleSetZ[photonInfo.x + photonID] = make_int2(splitZ, ~splitZ);
            }
        }
        */
    }

    /*** === OLD CODE === *****/

    /**
     * Computes the splitting planes photon bitmaps for the left side
     * and right side.
     *
     * The nodes can at most hold 32 photons.
     *
     * A thread for each photon in the nodes.
     */
    /*
    __global__ void OldPreprocessLowerNodes(int2 *splitTriangleSetX,
                                            int2 *splitTriangleSetY,
                                            int2 *splitTriangleSetZ,
                                            int *photonIndices,
                                            int *photonSets,
                                            point *photonPos,
                                            int nodeRange) {

        // Assumes 512 threads, just for the heck of it. Can be
        // templated at some point.
        
        __shared__ point sharedPos[128];

        int id = blockDim.x * blockIdx.x + threadIdx.x;
        unsigned int stepSize = gridDim.x * blockDim.x;
        int nodeID = id / 32;
        while (nodeID < nodeRange){

            int photonID = id - nodeID * 32;
            int photonSet = photonSets[nodeID];
            unsigned int* photonIndex = photonIndices[nodeID];

            // Just do the calculations for the 32 next photons. If
            // they are out of range noone cares and a guard would
            // only slow down execution.
            
            float3 pos = sharedPos[threadIdx.x] = photonPos[photonIndex + photonID];
            
            __syncthreads();

            int splitIndex = DIMENSION * (photonIndex + photonID);

            unsigned int splitX = 0;
            unsigned int splitY = 0;
            unsigned int splitZ = 0;

            #pragma unroll
            for (int i = 0; i < 32; ++i){
                splitX += 
            }
            
            
            nodeID += stepSize;
        }
        
    }
    */

}
}
}
}
