// Photon class for CUDA
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Scene/PhotonNode.h>

#include <Math/RandomGenerator.h>
#include <Resources/IDataBlock.h>
#include <Utils/CUDA/Convert.h>
#include <Utils/CUDA/Utils.h>

#include <sstream>

using namespace OpenEngine::Resources;

namespace OpenEngine {
    namespace Scene {

        PhotonNode::PhotonNode()
            : pos(NULL), maxSize(0), size(0) {}

        PhotonNode::PhotonNode(unsigned int size) 
            : maxSize(size), size(0) {
            logger.info << "Photon max: " << size << logger.end;
            cudaSafeMalloc(&pos, maxSize * sizeof(point));
            CHECK_FOR_CUDA_ERROR();
        }

        void PhotonNode::CreateRandomData(){
            point hat[maxSize];
            Math::RandomGenerator rand;
            //rand.SeedWithTime();
            for (unsigned int i = 0; i < maxSize; ++i)

                hat[i] = make_point(rand.UniformFloat(0.0f, 10.0f),
                                    rand.UniformFloat(0.0f, 10.0f),
                                    rand.UniformFloat(0.0f, 10.0f));
                /*
                hat[i] = make_point(rand.UniformInt(0.0f, 50.0f),
                                    rand.UniformInt(0.0f, 50.0f),
                                    rand.UniformInt(0.0f, 50.0f));
                */
            
            cudaMemcpy(pos, hat, maxSize * sizeof(point), cudaMemcpyHostToDevice);
            size = maxSize;
        }

        std::string PhotonNode::PositionToString(unsigned int begin, unsigned int range){
            std::ostringstream out;
            
            point position[range];
            cudaMemcpy(position, pos + begin, range * sizeof(point), cudaMemcpyDeviceToHost);
            CHECK_FOR_CUDA_ERROR();
            
            out << "[ 0: " << Utils::CUDA::Convert::ToString(position[0]);
            for (unsigned int i = 1; i < range; ++i){
                out << "\n " << i << ": " << Utils::CUDA::Convert::ToString(position[i]);
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
            
                point* verts;
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
                cudaMemcpy(position->GetVoidDataPtr(), pos, size * sizeof(point), cudaMemcpyDeviceToHost);
            }
        }
        
    }
}
