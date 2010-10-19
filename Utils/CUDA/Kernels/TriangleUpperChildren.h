// Kernels for creating triangle upper nodes children and splitting the triangles
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Meta/CUDA.h>
#include <Utils/CUDA/Kernels/PhotonMapDeviceVars.h>
#include <Scene/TriangleLowerNode.h>
#include <Utils/CUDA/Segments.h>

namespace OpenEngine {
    using namespace Scene;
namespace Utils {
namespace CUDA {
namespace Kernels {

    
    __global__ void 
    __launch_bounds__(Segments::SEGMENT_SIZE) 
        SetSplitSide(int2 *segmentPrimInfo,
                     int* segmentOwner,
                     char* info,
                     float *splitPoss,
                     float4* aabbMins,
                     float4* aabbMaxs,
                     int* splitSides){
        
        const int segmentID = blockIdx.x;

        if (segmentID < d_segments){
            int2 primInfo = segmentPrimInfo[segmentID];
            if (threadIdx.x < primInfo.y){
                int owner = segmentOwner[segmentID];
                char axis = info[owner];
                float splitPos = splitPoss[owner];
                
                const int id = primInfo.x + threadIdx.x;
                
                const float4 aabbMin = aabbMins[id];
                float leftPos = axis == KDNode::X ? aabbMin.x : aabbMin.y;
                leftPos = axis == KDNode::Z ? aabbMin.z : leftPos;
                splitSides[id] = leftPos < splitPos;

                const float4 aabbMax = aabbMaxs[id];
                float rightPos = axis == KDNode::X ? aabbMax.x : aabbMax.y;
                rightPos = axis == KDNode::Z ? aabbMax.z : rightPos;
                splitSides[id + d_triangles] = splitPos < rightPos;
            }
        }
    }

    __global__ void CalcNodeChildSize(int2* primitiveInfo,
                                      int *splitAddrs,
                                      int2* childSize){

        const int id = blockDim.x * blockIdx.x + threadIdx.x;
        
        if (id < d_activeNodeRange){
            const int2 primInfo = primitiveInfo[id];

            const int leftSize = (splitAddrs[primInfo.x + primInfo.y] - splitAddrs[primInfo.x]);
            const int rightSize = (splitAddrs[primInfo.x + primInfo.y + d_triangles] - splitAddrs[primInfo.x + d_triangles]);

            childSize[id] = make_int2(leftSize, rightSize);

            if (leftSize < TriangleLowerNode::MAX_SIZE ||
                rightSize < TriangleLowerNode::MAX_SIZE) d_createdLeafs = true;
        }
    }

    __global__ void 
    __launch_bounds__(Segments::SEGMENT_SIZE) 
        SetPrimitiveLeafSide(int2* segmentPrimInfo,
                             int *segmentOwner,
                             int2 *childSize,
                             int *splitSide,
                             int *leafSide){
        
        const int segmentID = blockIdx.x;

        if (segmentID < d_segments){
            int2 primInfo = segmentPrimInfo[segmentID];
            if (threadIdx.x < primInfo.y){
                const int owner = segmentOwner[segmentID];
                const int2 size = childSize[owner - d_activeNodeIndex];

                const int id = primInfo.x + threadIdx.x;
                
                leafSide[id] = (size.x < TriangleLowerNode::MAX_SIZE) * splitSide[id];
                leafSide[id + d_triangles] = (size.y < TriangleLowerNode::MAX_SIZE) * splitSide[id + d_triangles];
            }
        }
    }

    // @OPT include in CalcNodeChildSize? Wait until we construct the
    // entire tree to test this.
    __global__ void MarkNodeLeafs(int2 *childSize,
                                  int *leafNodes){

        const int id = blockDim.x * blockIdx.x + threadIdx.x;
        
        if (id < d_activeNodeRange){
            const int2 size = childSize[id];

            leafNodes[id] = size.x < TriangleLowerNode::MAX_SIZE;
            leafNodes[id + d_activeNodeRange] = size.y < TriangleLowerNode::MAX_SIZE;
        }
    }

    __global__ void CreateChildren(int2 *primitiveInfo,
                                   int2 *childSize,
                                   int *splitAddrs,
                                   int *left, int *right){
        const int id = blockDim.x * blockIdx.x + threadIdx.x;

        if (id < d_activeNodeRange){
            const int2 primInfo = primitiveInfo[id];
            
            const int leftID = id + d_activeNodeRange;
            const int rightID = leftID + d_activeNodeRange;

            const int2 size = childSize[id];
            const int leftIndex = splitAddrs[primInfo.x];
            const int rightIndex = splitAddrs[primInfo.x + d_triangles];

            primitiveInfo[leftID] = make_int2(leftIndex, size.x);
            primitiveInfo[rightID] = make_int2(rightIndex, size.y);

            left[id] = leftID + d_activeNodeIndex;
            right[id] = rightID + d_activeNodeIndex;
        }
    }

    __global__ void CreateChildren(int2 *primitiveInfo,
                                   int2 *childSize,
                                   int *splitAddrs,
                                   int *leafAddrs,
                                   int *leafNodeAddrs,
                                   int *left, int *right,
                                   int upperLeafPrimitives){

        int id = blockDim.x * blockIdx.x + threadIdx.x;

        if (id < d_activeNodeRange){
            
            const int2 primInfo = primitiveInfo[id];
            
            const int2 size = childSize[id];

            // @OPT move to constant memory
            const int totalLeafNodes = leafNodeAddrs[d_activeNodeRange * 2];
            
            bool isLeaf = size.x < TriangleLowerNode::MAX_SIZE;
            int leafAddr = leafNodeAddrs[id];
            int nonLeafAddr = id - leafAddr + totalLeafNodes;
            const int leftID = (isLeaf ? leafAddr : nonLeafAddr) + d_activeNodeRange;
            const int leftIndex = isLeaf ? leafAddrs[primInfo.x] + upperLeafPrimitives : splitAddrs[primInfo.x] - leafAddrs[primInfo.x];
            
            primitiveInfo[leftID] = make_int2(leftIndex, size.x);
            left[id] = leftID + d_activeNodeIndex;

            id += d_activeNodeRange;

            isLeaf = size.y < TriangleLowerNode::MAX_SIZE;
            leafAddr = leafNodeAddrs[id];
            nonLeafAddr = id - leafAddr + totalLeafNodes;
            const int rightID = (isLeaf ? leafAddr : nonLeafAddr) + d_activeNodeRange;
            const int rightIndex = isLeaf ? leafAddrs[primInfo.x + d_triangles] + upperLeafPrimitives : splitAddrs[primInfo.x + d_triangles] - leafAddrs[primInfo.x + d_triangles];

            primitiveInfo[rightID] = make_int2(rightIndex, size.y);
            right[id - d_activeNodeRange] = rightID + d_activeNodeIndex;
        }
    }

    __global__ void 
    __launch_bounds__(Segments::SEGMENT_SIZE) 
        SplitTriangles(int2 *segmentPrimInfo,
                       int *segmentOwners,
                       char *info, float *splitPositions,
                       int *splitSides, int *splitAddrs, 
                       float4* oldAabbMin, float4* oldAabbMax,
                       float4* newAabbMin, float4* newAabbMax){

        const int segmentID = blockIdx.x;
        
        if (segmentID < d_segments){
            int2 primInfo = segmentPrimInfo[segmentID];
            if (threadIdx.x < primInfo.y){
                const int owner = segmentOwners[segmentID];
                const char axis = info[owner];
                const float splitPos = splitPositions[owner];

                int triangleID = threadIdx.x + primInfo.x;
                const float4 aabbMin = oldAabbMin[triangleID];
                const float4 aabbMax = oldAabbMax[triangleID];

                int left = splitSides[triangleID];
                if (left == 1){
                    int addr = splitAddrs[triangleID];
                    newAabbMin[addr] = aabbMin;
                    newAabbMax[addr] = make_float4(axis == KDNode::X ? min(splitPos, aabbMax.x) : aabbMax.x,
                                                   axis == KDNode::Y ? min(splitPos, aabbMax.y) : aabbMax.y,
                                                   axis == KDNode::Z ? min(splitPos, aabbMax.z) : aabbMax.z,
                                                   aabbMax.w);
                }

                int right = splitSides[triangleID + d_triangles];
                if (right == 1){
                    int addr = splitAddrs[triangleID + d_triangles];
                    newAabbMin[addr] = make_float4(axis == KDNode::X ? max(splitPos, aabbMin.x) : aabbMin.x,
                                                   axis == KDNode::Y ? max(splitPos, aabbMin.y) : aabbMin.y,
                                                   axis == KDNode::Z ? max(splitPos, aabbMin.z) : aabbMin.z,
                                                   aabbMin.w);
                    newAabbMax[addr] = aabbMax;
                }

            }
        }
    }

    __global__ void 
    __launch_bounds__(Segments::SEGMENT_SIZE) 
        SplitTriangles(int2 *segmentPrimInfo,
                       int *segmentOwners,
                       char *info, float *splitPositions,
                       int *splitSides, int *splitAddrs, 
                       int *leafSides, int *leafAddrs, 
                       float4* oldAabbMin, float4* oldAabbMax,
                       float4* newAabbMin, float4* newAabbMax,
                       float4* finalAabbMin, float4* finalAabbMax){

        const int segmentID = blockIdx.x;
        
        if (segmentID < d_segments){
            int2 primInfo = segmentPrimInfo[segmentID];
            if (threadIdx.x < primInfo.y){
                
                const int owner = segmentOwners[segmentID];
                const char axis = info[owner];
                const float splitPos = splitPositions[owner];

                int triangleID = threadIdx.x + primInfo.x;
                const float4 aabbMin = oldAabbMin[triangleID];
                const float4 aabbMax = oldAabbMax[triangleID];

                int left = splitSides[triangleID];
                if (left == 1){
                    int leaf = leafSides[triangleID];
                    float4* resAabbMin = leaf ? finalAabbMin : newAabbMin;
                    float4* resAabbMax = leaf ? finalAabbMax : newAabbMax;
                    int addr = leaf ? leafAddrs[triangleID] : splitAddrs[triangleID] - leafAddrs[triangleID];
                    resAabbMin[addr] = aabbMin;
                    resAabbMax[addr] = make_float4(axis == KDNode::X ? min(splitPos, aabbMax.x) : aabbMax.x,
                                                   axis == KDNode::Y ? min(splitPos, aabbMax.y) : aabbMax.y,
                                                   axis == KDNode::Z ? min(splitPos, aabbMax.z) : aabbMax.z,
                                                   aabbMax.w);
                }

                triangleID += d_triangles;
                int right = splitSides[triangleID];
                if (right == 1){
                    int leaf = leafSides[triangleID];
                    float4* resAabbMin = leaf ? finalAabbMin : newAabbMin;
                    float4* resAabbMax = leaf ? finalAabbMax : newAabbMax;
                    int addr = leaf ? leafAddrs[triangleID] : splitAddrs[triangleID] - leafAddrs[triangleID];
                    resAabbMin[addr] = make_float4(axis == KDNode::X ? max(splitPos, aabbMin.x) : aabbMin.x,
                                                   axis == KDNode::Y ? max(splitPos, aabbMin.y) : aabbMin.y,
                                                   axis == KDNode::Z ? max(splitPos, aabbMin.z) : aabbMin.z,
                                                   aabbMin.w);
                    resAabbMax[addr] = aabbMax;
                }
                
            }
        }
    }

    __global__ void MarkLeafNodes(int* leafList, 
                                  char *nodeInfo,
                                  int leafIndex, int newLeafNodes){
        
        const int id = blockDim.x * blockIdx.x + threadIdx.x;
        
        if (id < newLeafNodes){
            leafList[id] = leafIndex + id;
            nodeInfo[id] = KDNode::LEAF;
        }
    }

}
}
}
}
