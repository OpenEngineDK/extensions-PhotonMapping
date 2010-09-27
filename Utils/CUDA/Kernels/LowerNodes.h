// Preprocess Lower Nodes.
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Meta/CUDA.h>
#include <Scene/KDNode.h>
#include <Scene/PhotonLowerNode.h>
#include <Utils/CUDA/Kernels/PhotonMapDeviceVars.h>
#include <Utils/CUDA/Utils.h>

using namespace OpenEngine::Scene;

namespace OpenEngine {
namespace Utils {
namespace CUDA {
namespace Kernels {

#define MIN_VVH 32

    __global__ void CalcSimpleSplittingPlane(char *lowerInfo,
                                             float *lowerSplitPos,
                                             int2 *lowerPhotonInfo,
                                             int2 *splitTriangleSet, // {X, Y, Z}
                                             float4 *positions,
                                             int *isNotLeaf){

        const int id = blockDim.x * blockIdx.x + threadIdx.x;

        if (id < d_activeNodeRange) {
            const int2 photonInfo = lowerPhotonInfo[id];
            float bestRelation = PhotonLowerNode::MAX_SIZE;
            float splitPos;
            char axis;

            int leftSet;
            char planeID;
            for (int bitmap = photonInfo.y; bitmap > 0; bitmap -= 1<<planeID){
                planeID = __ffs(bitmap) - 1;
                int2 splitSet = splitTriangleSet[photonInfo.x + planeID + 0 * d_photonNodes];
                
                leftSet = photonInfo.y & splitSet.x;
                int rightSet = photonInfo.y & splitSet.y;
                int diff = rightSet - leftSet;

                if (diff < bestRelation){
                    bestRelation = diff;
                    axis = KDNode::X;
                    splitPos = positions[photonInfo.x + planeID].x;
                }
            }

            lowerInfo[id] = bitcount(leftSet) <= 2 ? KDNode::LEAF : axis;
            lowerSplitPos[id] = splitPos;
            isNotLeaf[id] = bitcount(leftSet) <= 2 ? 0 : 1;
        }
    }

    __device__ int CalcVVH(int2 photonInfo,
                           int2 splitSet, 
                           float4 *positions){
        
        int leftSet = splitSet.x & photonInfo.y;
        const int leftCount = bitcount(leftSet);

        char bit = __ffs(leftSet) - 1;
        float4 pos = positions[photonInfo.x + bit];
        float3 min, max;
        min = max = make_float3(pos);
        leftSet -= 1<<bit;
        while (leftSet > 0){
            bit = __ffs(leftSet) - 1;
            float4 pos = positions[photonInfo.x + bit];
            minCorner(pos, min, min);
            maxCorner(pos, max, max);
            leftSet -= 1<<bit;
        }
        const int leftVolume = (max.x - min.x + 2 * PhotonLowerNode::SEARCH_RADIUS) *
            (max.y - min.y + 2 * PhotonLowerNode::SEARCH_RADIUS) *
            (max.z - min.z + 2 * PhotonLowerNode::SEARCH_RADIUS);
        const int leftVVH = leftCount * leftVolume;
          

        int rightSet = splitSet.x & photonInfo.y;
        const int rightCount = bitcount(leftSet);

        bit = __ffs(rightSet) - 1;
        pos = positions[photonInfo.x + bit];
        min = max = make_float3(pos);
        rightSet -= 1<<bit;
        while (rightSet > 0){
            bit = __ffs(rightSet) - 1;
            float4 pos = positions[photonInfo.x + bit];
            minCorner(pos, min, min);
            maxCorner(pos, max, max);
            rightSet -= 1<<bit;
        }
        const int rightVolume = (max.x - min.x + 2 * PhotonLowerNode::SEARCH_RADIUS) *
            (max.y - min.y + 2 * PhotonLowerNode::SEARCH_RADIUS) *
            (max.z - min.z + 2 * PhotonLowerNode::SEARCH_RADIUS);
        const int rightVVH = rightCount * rightVolume;

        return leftVVH + rightVVH;
    }

    __global__ void CalcVVHSplittingPlane(char *lowerInfo,
                                       float *lowerSplitPos,
                                       int2 *lowerPhotonInfo,
                                       int2 *splitTriangleSet, // {X, Y, Z}
                                       float4 *positions,
                                       int *isNotLeaf){
        const int id = blockDim.x * blockIdx.x + threadIdx.x;

        if (id < d_activeNodeRange) {
            const int2 photonInfo = lowerPhotonInfo[id];
            int bestVVH = 1<<30;
            float splitPos;
            char axis;

//#pragma unroll (not gonna happen, do manually)
            for (char dim = KDNode::X; dim < KDNode::Z; ++dim){
                char plane;
                for (int bitmap = photonInfo.y; bitmap > 0; bitmap -= 1<<plane){
                    plane = __ffs(bitmap) - 1;
                    int2 splitSet = splitTriangleSet[photonInfo.x + plane + dim * d_photonNodes];
                    
                    int VVH = CalcVVH(photonInfo, splitSet, positions);
                    
                    if (VVH < bestVVH){
                        bestVVH = VVH;
                        axis = dim;
                        switch(dim){
                        case KDNode::X:
                            splitPos = positions[photonInfo.x + plane].x;
                            break;
                        case KDNode::Y:
                            splitPos = positions[photonInfo.x + plane].y;
                            break;
                        case KDNode::Z:
                            splitPos = positions[photonInfo.x + plane].z;
                            break;
                        }
                    }
                }

            }

            lowerInfo[id] = bestVVH < MIN_VVH ? KDNode::LEAF :axis;
            lowerSplitPos[id] = splitPos;
        }
    }

}
}
}
}
