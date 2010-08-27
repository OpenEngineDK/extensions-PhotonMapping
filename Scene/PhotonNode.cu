// Photon class for CUDA
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Scene/PhotonNode.h>

#include <Resources/IDataBlock.h>
#include <Utils/CUDA/Convert.h>

#include <sstream>

using namespace OpenEngine::Resources;

namespace OpenEngine {
    namespace Scene {

        std::string PhotonNode::PositionToString(unsigned int begin, unsigned int range){
            std::ostringstream out;
            
            float3 position[range];
            cudaMemcpy(position, pos + begin, range * sizeof(float3), cudaMemcpyDeviceToHost);
            CHECK_FOR_CUDA_ERROR();
            
            out << "[ 0: " << Utils::CUDA::Convert::ToString(position[0]);
            for (unsigned int i = 1; i < range; ++i){
                out << ", " << i << ": " << Utils::CUDA::Convert::ToString(position[i]);
            }
            out << "]";
            return out.str();
        }

        void PhotonNode::MapToDataBlocks(IDataBlock* position){
            if (position->GetID() > 0){
                cudaGraphicsResource* resource;
                cudaGraphicsGLRegisterBuffer(&resource, position->GetID(), cudaGraphicsMapFlagsWriteDiscard);
                CHECK_FOR_CUDA_ERROR();
                
                cudaGraphicsMapResources(1, &resource, 0);
                CHECK_FOR_CUDA_ERROR();
            
                float3* verts;
                size_t bytes;
                cudaGraphicsResourceGetMappedPointer((void**)&verts, &bytes,
                                                     resource);
                CHECK_FOR_CUDA_ERROR();
                
                cudaMemcpy(verts, pos, bytes, cudaMemcpyDeviceToDevice);
                CHECK_FOR_CUDA_ERROR();
                
                cudaGraphicsUnmapResources(1, &resource, 0);
                CHECK_FOR_CUDA_ERROR();
                
                cudaGraphicsUnregisterResource(resource);
                CHECK_FOR_CUDA_ERROR();
            }else if (position->GetVoidDataPtr() != NULL){
                cudaMemcpy(position->GetVoidDataPtr(), pos, size * sizeof(float3), cudaMemcpyDeviceToHost);
            }
        }
        
    }
}
