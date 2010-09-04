// KD tree upper node for photons
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Scene/KDPhotonUpperNode.h>
#include <Resources/IDataBlock.h>
#include <Utils/CUDA/Kernels/UpperNodeMapToGL.hcu>

#include <Logging/Logger.h>

#include <sstream>

using namespace OpenEngine::Resources;
using namespace OpenEngine::Utils::CUDA::Kernels;

namespace OpenEngine {
    namespace Scene {

        KDPhotonUpperNode::KDPhotonUpperNode()
            : maxSize(0), size(0){}

        KDPhotonUpperNode::KDPhotonUpperNode(unsigned int size)
            : maxSize(size), size(0) {

            cudaMalloc(&info, maxSize * sizeof(char));
            cudaMalloc(&splitPos, maxSize * sizeof(float));
            cudaMalloc(&aabbMin, maxSize * sizeof(point));
            cudaMalloc(&aabbMax, maxSize * sizeof(point));

            cudaMalloc(&photonIndex, maxSize * sizeof(unsigned int));
            cudaMalloc(&range, maxSize * sizeof(unsigned int));
            cudaMalloc(&parent, maxSize * sizeof(unsigned int));
            cudaMalloc(&child, maxSize * sizeof(unsigned int));

            CHECK_FOR_CUDA_ERROR();
        }

        void KDPhotonUpperNode::Resize(unsigned int i){
            unsigned int copySize = i < size ? i : size;
            
            char *tempChar;
            float *tempFloat;
            point *tempPoint;
            unsigned int *tempUint;

            cudaMalloc(&tempChar, i * sizeof(char));
            cudaMemcpy(tempChar, info, copySize * sizeof(char), cudaMemcpyDeviceToDevice);
            cudaFree(info);
            info = tempChar;
            CHECK_FOR_CUDA_ERROR();

            cudaMalloc(&tempFloat, i * sizeof(float));
            cudaMemcpy(tempFloat, splitPos, copySize * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaFree(splitPos);
            splitPos = tempFloat;
            CHECK_FOR_CUDA_ERROR();

            cudaMalloc(&tempPoint, i * sizeof(point));
            cudaMemcpy(tempPoint, aabbMin, copySize * sizeof(point), cudaMemcpyDeviceToDevice);
            cudaFree(aabbMin);
            aabbMin = tempPoint;
            CHECK_FOR_CUDA_ERROR();

            cudaMalloc(&tempPoint, i * sizeof(point));
            cudaMemcpy(tempPoint, aabbMax, copySize * sizeof(point), cudaMemcpyDeviceToDevice);
            cudaFree(aabbMax);
            aabbMax = tempPoint;
            CHECK_FOR_CUDA_ERROR();

            cudaMalloc(&tempUint, i * sizeof(unsigned int));
            cudaMemcpy(tempUint, photonIndex, copySize * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
            cudaFree(photonIndex);
            photonIndex = tempUint;
            CHECK_FOR_CUDA_ERROR();

            cudaMalloc(&tempUint, i * sizeof(unsigned int));
            cudaMemcpy(tempUint, range, copySize * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
            cudaFree(range);
            range = tempUint;
            CHECK_FOR_CUDA_ERROR();

            cudaMalloc(&tempUint, i * sizeof(unsigned int));
            cudaMemcpy(tempUint, parent, copySize * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
            cudaFree(parent);
            parent = tempUint;
            CHECK_FOR_CUDA_ERROR();

            cudaMalloc(&tempUint, i * sizeof(unsigned int));
            cudaMemcpy(tempUint, child, copySize * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
            cudaFree(child);
            child = tempUint;
            CHECK_FOR_CUDA_ERROR();

            maxSize = i;
        }

        void KDPhotonUpperNode::MapToDataBlocks(Resources::IDataBlock* position,
                                                Resources::IDataBlock* colors){
#ifdef DEBUG
            if (position->GetID() == 0 && colors->GetID() == 0)
                return;
#endif
            
            cudaGraphicsResource *pResource, *cResource;
            cudaGraphicsGLRegisterBuffer(&pResource, position->GetID(), cudaGraphicsMapFlagsWriteDiscard);
            cudaGraphicsGLRegisterBuffer(&cResource, colors->GetID(), cudaGraphicsMapFlagsWriteDiscard);
            CHECK_FOR_CUDA_ERROR();
            
            cudaGraphicsMapResources(1, &pResource, 0);
            cudaGraphicsMapResources(1, &cResource, 0);
            CHECK_FOR_CUDA_ERROR();

            float3* posv;
            size_t bytes;
            cudaGraphicsResourceGetMappedPointer((void**)&posv, &bytes,
                                                 pResource);
            float3* colv;
            cudaGraphicsResourceGetMappedPointer((void**)&colv, &bytes,
                                                 cResource);
            CHECK_FOR_CUDA_ERROR();

            unsigned int s = min(size, position->GetSize());
            
            UpperNodeMapToGL<<<64, 128>>>(*this, posv, colv, s);
            CHECK_FOR_CUDA_ERROR();

            cudaGraphicsUnmapResources(1, &pResource, 0);
            cudaGraphicsUnmapResources(1, &cResource, 0);
            CHECK_FOR_CUDA_ERROR();
            
            cudaGraphicsUnregisterResource(pResource);
            cudaGraphicsUnregisterResource(cResource);
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
                    
            point h_aabbmin, h_aabbmax;
            cudaMemcpy(&h_aabbmin, aabbMin + i, sizeof(point), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_aabbmax, aabbMax + i, sizeof(point), cudaMemcpyDeviceToHost);
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
            
            point pos[size];
            cudaMemcpy(&pos, photons.pos+index, 
                       size*sizeof(point), cudaMemcpyDeviceToHost);
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
                    
            point positions[photonRange];
            cudaMemcpy(positions, photons.pos + photonStart, photonRange * sizeof(point), cudaMemcpyDeviceToHost);
            point hostMax = positions[0];
            point hostMin = positions[0];
            for (unsigned int p = 1; p < photonRange; ++p){
                hostMax = make_float3(max(hostMax.x, positions[p].x),
                                      max(hostMax.y, positions[p].y),
                                      max(hostMax.z, positions[p].z));
                hostMin = make_float3(min(hostMin.x, positions[p].x),
                                      min(hostMin.y, positions[p].y),
                                      min(hostMin.z, positions[p].z));
            }
                    
            point gpuMax;
            point gpuMin;
            cudaMemcpy(&gpuMax, aabbMax + i, sizeof(point), cudaMemcpyDeviceToHost);
            cudaMemcpy(&gpuMin, aabbMin + i, sizeof(point), cudaMemcpyDeviceToHost);
    
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
