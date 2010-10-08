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
                
                if (lowerNodes->maxSize < upperNodeLeafList.size)
                    lowerNodes->Resize(upperNodeLeafList.size);

                unsigned int blocks, threads;
                Calc1DKernelDimensions(upperNodeLeafList.size, blocks, threads);
                START_TIMER(timerID);
                CreateLowerNodes<<<blocks, threads>>>(upperNodeLeafList.leafIDs,
                                                      upperNodes->aabbMin->GetDeviceData(),
                                                      upperNodes->aabbMax->GetDeviceData(),
                                                      upperNodes->photonInfo->GetDeviceData(),
                                                      upperNodes->GetLeftData(),
                                                      upperNodes->GetRightData(),
                                                      lowerNodes->info->GetDeviceData(),
                                                      lowerNodes->photonInfo->GetDeviceData(),
                                                      lowerNodes->extendedVolume,
                                                      upperNodeLeafList.size);
                PRINT_TIMER(timerID, "Create Lower Nodes");
                CHECK_FOR_CUDA_ERROR();
                lowerNodes->size = upperNodeLeafList.size;

                // Calculate all splitting planes.
                Calc1DKernelDimensions(lowerNodes->size * 32, blocks, threads);
                int smemSize = 512 * sizeof(float4);
                START_TIMER(timerID);
                CreateSplittingPlanes<<<blocks, threads, smemSize>>>
                    (lowerNodes->splitTriangleSetX,
                     lowerNodes->splitTriangleSetY,
                     lowerNodes->splitTriangleSetZ,
                     lowerNodes->photonInfo->GetDeviceData(),
                     photons.pos,
                     lowerNodes->size);
                PRINT_TIMER(timerID, "Splitting plane creation");
                CHECK_FOR_CUDA_ERROR();
            }
            
            void PhotonMap::ProcessLowerNodes(int activeIndex,
                                              int activeRange,
                                              int &leafsCreated,
                                              int &childrenCreated){
                
                logger.info << "=== Process " << activeRange << " Lower Nodes Starting at " << activeIndex << " ===" << logger.end;

                // Check that there is room for the new children
                if (lowerNodes->maxSize < lowerNodes->size + activeRange * 2)
                    lowerNodes->Resize(lowerNodes->size + activeRange * 2);

                cudaMemcpyToSymbol(d_activeNodeRange, &activeRange, sizeof(int));

                // @OPT mark leafs and only do scan if there actually
                // are any. Otherwise just copy.

                // Compute split plane and mark nodes with low split
                // cost as leaf nodes
                unsigned int blocks, threads;
                Calc1DKernelDimensions(activeRange, blocks, threads);
                START_TIMER(timerID);
                CalcSimpleSplittingPlane<<<blocks, threads>>>(lowerNodes->info->GetDeviceData() + activeIndex,
                                                              lowerNodes->splitPos->GetDeviceData() + activeIndex,
                                                              lowerNodes->photonInfo->GetDeviceData() + activeIndex,
                                                              lowerNodes->splitTriangleSet,
                                                              photons.pos,
                                                              leafSide);
                PRINT_TIMER(timerID, "Lower node splitting plane calculation");
                
                // Calculate child indexes
                cudppScan(scanHandle, leafPrefix, leafSide, activeRange+1);

                // Split nodes to children
                

                leafsCreated = childrenCreated = 0;

            }
            
        }
    }
}
