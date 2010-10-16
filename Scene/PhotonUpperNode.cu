// KD tree upper node for photons
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Scene/PhotonUpperNode.h>
#include <Resources/IDataBlock.h>
#include <Utils/CUDA/Kernels/UpperNodeMapToGL.hcu>
#include <Utils/CUDA/Utils.h>

#include <Logging/Logger.h>

using namespace OpenEngine::Resources;
using namespace OpenEngine::Utils::CUDA::Kernels;

namespace OpenEngine {
    namespace Scene {

        PhotonUpperNode::PhotonUpperNode()
            : KDNode() {}

        PhotonUpperNode::PhotonUpperNode(int size)
            : KDNode(size) {

            logger.info << "Photon upper node inital max: " << size<< logger.end;

            //cudaSafeMalloc(&parents, maxSize * sizeof(int));

            CHECK_FOR_CUDA_ERROR();
        }

        void PhotonUpperNode::Resize(int i){
            KDNode::Resize(i);
            /*            
            unsigned int copySize = this->size;
            
            int *tempInt;
            cudaSafeMalloc(&tempInt, i * sizeof(int));
            cudaMemcpy(tempInt, parents, copySize * sizeof(int), cudaMemcpyDeviceToDevice);
            cudaFree(parents);
            parents = tempInt;
            CHECK_FOR_CUDA_ERROR();

            maxSize = i;
            size = copySize;
            */
        }

        void PhotonUpperNode::MapToDataBlocks(Resources::IDataBlock* position,
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

            unsigned int blocks, threads;
            Calc1DKernelDimensions(s, blocks, threads);
            //logger.info << "blocks " << blocks << ", threads " << threads << logger.end;
            UpperNodeMapToGL<<<blocks, threads/2>>>(aabbMin->GetDeviceData(), aabbMax->GetDeviceData(), splitPos->GetDeviceData(), info->GetDeviceData(), posv, colv, s);
            CHECK_FOR_CUDA_ERROR();

            cudaGraphicsUnmapResources(1, &pResource, 0);
            cudaGraphicsUnmapResources(1, &cResource, 0);
            CHECK_FOR_CUDA_ERROR();
            
            cudaGraphicsUnregisterResource(pResource);
            cudaGraphicsUnregisterResource(cResource);
            CHECK_FOR_CUDA_ERROR();
        }
                
        std::string PhotonUpperNode::PhotonsToString(unsigned int i, 
                                                     PhotonNode photons){
            std::ostringstream out;

            int2 info;
            cudaMemcpy(&info, photonInfo+i, sizeof(int2), cudaMemcpyDeviceToHost);

            point pos[info.y];
            cudaMemcpy(&pos, photons.pos+info.x, 
                       info.y*sizeof(point), cudaMemcpyDeviceToHost);
            CHECK_FOR_CUDA_ERROR();
            
            out << Utils::CUDA::Convert::ToString(pos[0]);
            for (int i = 1; i < info.y; ++i){
                out << "\n" << Utils::CUDA::Convert::ToString(pos[i]);
            }

            return out.str();
        }

    }
}
