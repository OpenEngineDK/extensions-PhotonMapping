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
#include <Utils/PhotonMapUtils.h>

#include <Utils/CUDA/Kernels/PhotonMapDeviceVars.h>
#include <Utils/CUDA/Kernels/PreprocessLowerNodes.h>
#include <Utils/CUDA/Kernels/LowerNodes.h>

#include <Logging/Logger.h>

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {

            using namespace Kernels;

            void PhotonMap::PreprocessLowerNodes(){

                logger.info << "Preprocess " << upperNodeLeafList.size << " lower nodes" << logger.end;
                
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

                // Calculate all splitting planes.
                Calc1DKernelDimensions(lowerNodes.size * 32, blocks, threads);
                int smemSize = 512 * sizeof(float4);
                START_TIMER(timerID);
                CreateSplittingPlanes<<<blocks, threads, smemSize>>>
                    (lowerNodes.splitTriangleSetX,
                     lowerNodes.splitTriangleSetY,
                     lowerNodes.splitTriangleSetZ,
                     lowerNodes.photonInfo,
                     photons.pos,
                     lowerNodes.size);
                CHECK_FOR_CUDA_ERROR();
                PRINT_TIMER(timerID, "Splitting plane creation");
            }
            
            void PhotonMap::ProcessLowerNodes(int activeIndex,
                                              int activeRange,
                                              int &leafsCreated,
                                              int &childrenCreated){
                
                logger.info << "=== Process " << activeRange << " Lower Nodes Starting at " << activeIndex << " ===" << logger.end;

                // Check that there is room for the new children
                if (lowerNodes.maxSize < lowerNodes.size + activeRange * 2)
                    lowerNodes.Resize(lowerNodes.size + activeRange * 2);

                cudaMemcpyToSymbol(d_activeNodeRange, &activeRange, sizeof(int));

                // Compute split plane and mark nodes with low split
                // cost as leaf nodes
                unsigned int blocks, threads;
                Calc1DKernelDimensions(activeRange, blocks, threads);
                CalcVVHSplittingPlane<<<blocks, threads>>>(lowerNodes.info + activeIndex,
                                                              lowerNodes.splitPos + activeIndex,
                                                              lowerNodes.photonInfo + activeIndex,
                                                              lowerNodes.splitTriangleSet,
                                                              photons.pos,
                                                              leafSide);
                
                // Calculate child indexes
                cudppScan(scanHandle, leafPrefix, leafSide, activeRange);

                // Split nodes to children

                leafsCreated = childrenCreated = 0;

            }
            
        }
    }
}
