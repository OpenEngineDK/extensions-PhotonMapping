// Photon map class for CUDA
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Utils/CUDA/PhotonMap.h>
#include <Utils/CUDA/Utils.h>

#include <Utils/CUDA/Kernels/PreprocessLowerNodes.h>

#include <Logging/Logger.h>

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {

            using namespace Kernels;

            void PhotonMap::PreprocessLowerNodes(){

                // Create lower nodes and their splitting planes.
                //cudaMemcpyToSymbol(d_upperPhotonInfo, &(upperNodes.photonInfo), sizeof(upperNodes.photonInfo));
                //cudaMemcpyToSymbol(d_lowerPhotonInfo, &(lowerNodes.photonInfo), sizeof(lowerNodes.photonInfo));
                //CHECK_FOR_CUDA_ERROR();

                if (lowerNodes.maxSize < upperNodeLeafList.size)
                    lowerNodes.Resize(upperNodeLeafList.size);

                unsigned int blocks, threads;
                Calc1DKernelDimensions(upperNodeLeafList.size, blocks, threads);
                CreateLowerNodes<<<blocks, threads>>>(upperNodeLeafList.leafIDs,
                                                      upperNodes.photonInfo,
                                                      lowerNodes.info,
                                                      lowerNodes.photonInfo,
                                                      upperNodeLeafList.size);
                CHECK_FOR_CUDA_ERROR();
                lowerNodes.size = upperNodeLeafList.size;

                //logger.info << "Leaf IDs: " << Utils::CUDA::Convert::ToString(upperNodeLeafList.leafIDs, lowerNodes.size) << logger.end;
                //logger.info << "Lower node photon info: " << Utils::CUDA::Convert::ToString(lowerNodes.photonInfo, lowerNodes.size) << logger.end;

                // Calculate all splitting planes.
                Calc1DKernelDimensions(lowerNodes.size * 32, blocks, threads);
                int smemSize = 512 * sizeof(float4);
                //logger.info << "<<<" << blocks <<  ", " << threads << ", " << smemSize << ">>>" << logger.end;
                //START_TIMER(timerID);
                CreateSplittingPlanes<<<blocks, threads, smemSize>>>
                    (lowerNodes.splitTriangleSetX,
                     lowerNodes.splitTriangleSetY,
                     lowerNodes.splitTriangleSetZ,
                     lowerNodes.photonInfo,
                     photons.pos,
                     lowerNodes.size);
                CHECK_FOR_CUDA_ERROR();
                
                //PRINT_TIMER(timerID, "Splitting plane creation");
            }
            
            void PhotonMap::ProcessLowerNodes(int activeIndex,
                                              int activeRange,
                                              int &childrenCreated){

            }
            
        }
    }
}
