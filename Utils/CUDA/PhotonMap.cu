// Photon map class for CUDA
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Utils/CUDA/PhotonKDTree.h>

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {

            PhotonMap::PhotonMap(unsigned int size) {
                MAX_BLOCKS = (activeCudaDevice.maxGridSize[0] + 1) / activeCudaDevice.maxThreadsDim[0];
                logger.info << "MAX_BLOCKS " << MAX_BLOCKS << logger.end;

                // Initialized timer
                cutCreateTimer(&timerID);

                // Allocate photons on GPU
                photons = PhotonNode(size);
                photons.CreateRandomData();

                // Make room for the root node
                upperNodes = KDPhotonUpperNode(1);

                // Split vars
                splitVars.Init(size);
                scanConfig.op = CUDPP_ADD;
                scanConfig.datatype = CUDPP_UINT;
                scanConfig.algorithm = CUDPP_SCAN;
                scanConfig.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
                
                CUDPPResult res = cudppPlan(&scanHandle, scanConfig, size, 1, 0);
                
                if (CUDPP_SUCCESS != res)
                    throw Core::Exception("Error creating CUDPP scanPlan");
            }

            void PhotonMap::Create(){

            }

            void PhotonMap::ProcessUpperNodes(unsigned int activeIndex,
                                              unsigned int activeRange,
                                              unsigned int &childrenCreated){

            }
            
            void PhotonMap::ComputeBoundingBox(unsigned int activeIndex,
                                               unsigned int activeRange){

            }
            
            void PhotonMap::SplitUpperNodePhotons(unsigned int activeIndex,
                                                  unsigned int activeRange){

            }
            
            unsigned int PhotonMap::CreateChildren(unsigned int activeIndex,
                                                   unsigned int activeRange){
                return activeRange * 2;
            }
            
            void PhotonMap::PreprocessLowerNodes(unsigned int range){

            }
            
            void PhotonMap::ProcessLowerNodes(unsigned int activeIndex,
                                              unsigned int activeRange,
                                              unsigned int &childrenCreated){

            }
            
        }
    }
}
