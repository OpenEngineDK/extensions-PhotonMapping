// KD tree upper node for photons
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Scene/KDPhotonUpperNode.h>
#include <sstream>
#include <Logging/Logger.h>

namespace OpenEngine {
    namespace Scene {

        void KDPhotonUpperNode::Init(unsigned int size){
            maxSize = size;
            cudaMalloc(&info, maxSize * sizeof(char));
            cudaMalloc(&splitPos, maxSize * sizeof(float));
            cudaMalloc(&aabbMin, maxSize * sizeof(float3));
            cudaMalloc(&aabbMax, maxSize * sizeof(float3));
            cudaMalloc(&photonIndex, maxSize * sizeof(unsigned int));
            cudaMalloc(&range, maxSize * sizeof(unsigned int));
            cudaMalloc(&parent, maxSize * sizeof(unsigned int));
            cudaMalloc(&child, maxSize * sizeof(unsigned int));
            CHECK_FOR_CUDA_ERROR();
        }
                
        std::string KDPhotonUpperNode::ToString(unsigned int i){
            bool isLeaf = false;
            std::ostringstream out;
                    
            out << "Upper node " << i << ":\n";
            char h_info;
            cudaMemcpy(&h_info, info + i, sizeof(char), cudaMemcpyDeviceToHost);
            CHECK_FOR_CUDA_ERROR();
                    
            float h_pos;
            cudaMemcpy(&h_pos, splitPos + i, sizeof(float), cudaMemcpyDeviceToHost);
            CHECK_FOR_CUDA_ERROR();
            switch(h_info){
            case X:
                out << "Splits along the X plane at pos " << h_pos << "\n";
                break;
            case Y:
                out << "Splits along the Y plane at pos " << h_pos << "\n";
                break;
            case Z:
                out << "Splits along the Z plane at pos " << h_pos << "\n";
                break;
            case LEAF:
                isLeaf = true;
                out << "Is a leaf\n";
                break;
            }

            unsigned int h_index, h_range;
            cudaMemcpy(&h_index, photonIndex + i, sizeof(unsigned int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_range, range + i, sizeof(unsigned int), cudaMemcpyDeviceToHost);
            out << "Index " << h_index << " and range " << h_range << "\n";
                    
            float3 h_aabbmin, h_aabbmax;
            cudaMemcpy(&h_aabbmin, aabbMin + i, sizeof(float3), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_aabbmax, aabbMax + i, sizeof(float3), cudaMemcpyDeviceToHost);
            CHECK_FOR_CUDA_ERROR();
            out << "Axis aligned bounding box: " << Utils::CUDA::Convert::ToString(h_aabbmin);
            out << " -> " << Utils::CUDA::Convert::ToString(h_aabbmax) << "\n";
                    
            if (i != 0){
                unsigned int p;
                cudaMemcpy(&p, parent + i, sizeof(unsigned int), cudaMemcpyDeviceToHost);
                out << "Has parent " << p << " and ";
            }
                    
            unsigned int h_child;
            cudaMemcpy(&h_child, child + i, sizeof(unsigned int), cudaMemcpyDeviceToHost);
            CHECK_FOR_CUDA_ERROR();
            if (!isLeaf){
                out << "has leftchild " << h_child << "\n";
            }else{
                out << "points to lowernode " << h_child << "\n";
            }
                    
            return out.str();
        }

        std::string KDPhotonUpperNode::PhotonsToString(unsigned int i, 
                                                       PhotonNode photons){
            std::ostringstream out;
            
            unsigned int index,size;
            cudaMemcpy(&index, photonIndex+i, sizeof(unsigned int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&size, range+i, sizeof(unsigned int), cudaMemcpyDeviceToHost);
            
            float3 pos[size];
            cudaMemcpy(&pos, photons.pos+index, 
                       size*sizeof(float3), cudaMemcpyDeviceToHost);
            CHECK_FOR_CUDA_ERROR();
            
            out << Utils::CUDA::Convert::ToString(pos[0]);
            for (unsigned int i = 1; i < size; ++i){
                out << "\n" << Utils::CUDA::Convert::ToString(pos[i]);
            }

            return out.str();
        }

        void KDPhotonUpperNode::CheckBoundingBox(unsigned int i, PhotonNode photons){
            unsigned int photonStart;
            cudaMemcpy(&photonStart, photonIndex + i, sizeof(unsigned int), cudaMemcpyDeviceToHost);
            unsigned int photonRange;
            cudaMemcpy(&photonRange, range + i, sizeof(unsigned int), cudaMemcpyDeviceToHost);
                    
            float3 positions[photonRange];
            cudaMemcpy(positions, photons.pos + photonStart, photonRange * sizeof(float3), cudaMemcpyDeviceToHost);
            float3 hostMax = positions[0];
            float3 hostMin = positions[0];
            for (unsigned int p = 1; p < photonRange; ++p){
                hostMax = make_float3(max(hostMax.x, positions[p].x),
                                      max(hostMax.y, positions[p].y),
                                      max(hostMax.z, positions[p].z));
                hostMin = make_float3(min(hostMin.x, positions[p].x),
                                      min(hostMin.y, positions[p].y),
                                      min(hostMin.z, positions[p].z));
            }
                    
            float3 gpuMax;
            float3 gpuMin;
            cudaMemcpy(&gpuMax, aabbMax + i, sizeof(float3), cudaMemcpyDeviceToHost);
            cudaMemcpy(&gpuMin, aabbMin + i, sizeof(float3), cudaMemcpyDeviceToHost);
    
            if (hostMax.x != gpuMax.x || hostMax.y != gpuMax.y || hostMax.z != gpuMax.z){
                logger.info << "CPU max " << Utils::CUDA::Convert::ToString(hostMax);
                logger.info << " != GPU max " << Utils::CUDA::Convert::ToString(gpuMax) << logger.end;
            }
                    
            if (hostMin.x != gpuMin.x || hostMin.y != gpuMin.y || hostMin.z != gpuMin.z){
                logger.info << "CPU min " << Utils::CUDA::Convert::ToString(hostMin);
                logger.info << " != GPU min " << Utils::CUDA::Convert::ToString(gpuMin) << logger.end;
            }

        }

    }
}
