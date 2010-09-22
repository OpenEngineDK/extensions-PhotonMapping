// Photon map class for CUDA
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Utils/CUDA/PhotonMap.h>
#include <Utils/CUDA/Point.h>
#include <Utils/CUDA/Utils.h>

#include <Core/Exceptions.h>

using namespace OpenEngine::Core;

#include <Logging/Logger.h>

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {
            
            void PhotonMap::VerifyMap(){
                char info;
                cudaMemcpy(&info, upperNodes.info, sizeof(char), cudaMemcpyDeviceToHost);

                float splitPos;
                cudaMemcpy(&splitPos, upperNodes.splitPos, sizeof(float), cudaMemcpyDeviceToHost);

                point aabbMin, aabbMax;
                cudaMemcpy(&aabbMin, upperNodes.aabbMin, sizeof(point), cudaMemcpyDeviceToHost);
                cudaMemcpy(&aabbMax, upperNodes.aabbMax, sizeof(point), cudaMemcpyDeviceToHost);

                point aabbMinAdjusted = aabbMin, aabbMaxAdjusted = aabbMax;
                switch(info){
                case KDNode::X:
                    aabbMinAdjusted.x = aabbMaxAdjusted.x = splitPos;
                    break;
                case KDNode::Y:
                    aabbMinAdjusted.y = aabbMaxAdjusted.y = splitPos;
                    break;
                case KDNode::Z:
                    aabbMinAdjusted.z = aabbMaxAdjusted.z = splitPos;
                    break;
                }

                int left, right;
                cudaMemcpy(&left, upperNodes.left, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&right, upperNodes.right, sizeof(int), cudaMemcpyDeviceToHost);

                int leftSize = VerifyUpperNode(left, info, splitPos, aabbMin, aabbMaxAdjusted);
                int rightSize = VerifyUpperNode(right, info, splitPos, aabbMinAdjusted, aabbMax);

                int2 photonInfo;
                cudaMemcpy(&photonInfo, upperNodes.photonInfo, sizeof(int2), cudaMemcpyDeviceToHost);

                if (leftSize + rightSize != photonInfo.y)
                    throw Exception("Root nodes size " + 
                                    Utils::Convert::ToString(photonInfo.y) + 
                                    " isn't the sum of left size " + 
                                    Utils::Convert::ToString(leftSize) +
                                    " and right size " + Utils::Convert::ToString(rightSize));
            }

            int PhotonMap::VerifyUpperNode(int index, char parentInfo, float parentSplitPos,
                                           point parentAABBMin, point parentAABBMax){
                char info;
                cudaMemcpy(&info, upperNodes.info+index, sizeof(char), cudaMemcpyDeviceToHost);
                
                int2 photonInfo;
                cudaMemcpy(&photonInfo, upperNodes.photonInfo+index, sizeof(int2), cudaMemcpyDeviceToHost);
                    
                if (info == KDNode::LEAF){
                    // Base case
                    
                    point positions[photonInfo.y];
                    cudaMemcpy(positions, photons.pos + photonInfo.x, 
                               photonInfo.y * sizeof(point), cudaMemcpyDeviceToHost);
                    point aabbMin, aabbMax;
                    aabbMin = aabbMax = positions[0];
                    for (int i = 1; i < photonInfo.y; ++i){
                        aabbMin = pointMin(aabbMin, positions[i]);
                        aabbMax = pointMax(aabbMax, positions[i]);
                    }

                    if (!aabbContains(parentAABBMin, parentAABBMax, aabbMin))
                        throw Exception("Leaf node " + Utils::Convert::ToString(index) +
                                        " aabb minimum cornor " + Utils::CUDA::Convert::ToString(aabbMin) +
                                        " is not contained in parents aabb " + Utils::CUDA::Convert::ToString(parentAABBMin) + 
                                        " -> " + Utils::CUDA::Convert::ToString(parentAABBMax) + ".");

                    if (!aabbContains(parentAABBMin, parentAABBMax, aabbMax))
                        throw Exception("Leaf node " + Utils::Convert::ToString(index) +
                                        " aabb maximum cornor " + Utils::CUDA::Convert::ToString(aabbMax) +
                                        " is not contained in parents aabb " + Utils::CUDA::Convert::ToString(parentAABBMin) + 
                                        " -> " + Utils::CUDA::Convert::ToString(parentAABBMax) + ".");
                    
                }else{
                    // Check parent info
                    float splitPos;
                    cudaMemcpy(&splitPos, upperNodes.splitPos+index, sizeof(float), cudaMemcpyDeviceToHost);
                    
                    point aabbMin, aabbMax;
                    cudaMemcpy(&aabbMin, upperNodes.aabbMin+index, sizeof(point), cudaMemcpyDeviceToHost);
                    cudaMemcpy(&aabbMax, upperNodes.aabbMax+index, sizeof(point), cudaMemcpyDeviceToHost);

                    if (!aabbContains(parentAABBMin, parentAABBMax, aabbMin))
                        throw Exception("Node " + Utils::Convert::ToString(index) +
                                        " aabb minimum cornor " + Utils::CUDA::Convert::ToString(aabbMin) +
                                        " is not contained in parents aabb " + Utils::CUDA::Convert::ToString(parentAABBMin) + 
                                        " -> " + Utils::CUDA::Convert::ToString(parentAABBMax) + ".");

                    if (!aabbContains(parentAABBMin, parentAABBMax, aabbMax))
                        throw Exception("Node " + Utils::Convert::ToString(index) +
                                        " aabb maximum cornor " + Utils::CUDA::Convert::ToString(aabbMax) +
                                        " is not contained in parents aabb " + Utils::CUDA::Convert::ToString(parentAABBMin) + 
                                        " -> " + Utils::CUDA::Convert::ToString(parentAABBMax) + ".");


                    point aabbMinAdjusted = aabbMin, aabbMaxAdjusted = aabbMax;
                    switch(info){
                    case KDNode::X:
                        aabbMinAdjusted.x = aabbMaxAdjusted.x = splitPos;
                        break;
                    case KDNode::Y:
                        aabbMinAdjusted.y = aabbMaxAdjusted.y = splitPos;
                        break;
                    case KDNode::Z:
                        aabbMinAdjusted.z = aabbMaxAdjusted.z = splitPos;
                        break;
                    }
                    
                    int left, right;
                    cudaMemcpy(&left, upperNodes.left+index, sizeof(int), cudaMemcpyDeviceToHost);
                    cudaMemcpy(&right, upperNodes.right+index, sizeof(int), cudaMemcpyDeviceToHost);

                    int leftSize = VerifyUpperNode(left, info, splitPos, aabbMin, aabbMaxAdjusted);
                    int rightSize = VerifyUpperNode(right, info, splitPos, aabbMinAdjusted, aabbMax);

                    if (leftSize + rightSize != photonInfo.y)
                        throw Exception("The " + Utils::Convert::ToString(index) + 
                                        "'th node's size " + 
                                        Utils::Convert::ToString(photonInfo.y) + 
                                        " isn't the sum of left size " + 
                                        Utils::Convert::ToString(leftSize) +
                                        " and right size " + Utils::Convert::ToString(rightSize));
                }

                return photonInfo.y;
            }

        }
    }
}
