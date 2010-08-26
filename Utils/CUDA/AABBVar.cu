// Variables for calculating Axis Aligned Bounding Boxes
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Utils/CUDA/AABBVar.h>

#include <sstream>

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {

            std::string AABBVar::MaxToString(unsigned int size) {
                std::ostringstream out;
                
                float3 hostMax[size];
                cudaMemcpy(hostMax, max, size * sizeof(float3), cudaMemcpyDeviceToHost);
                CHECK_FOR_CUDA_ERROR();
                
                out << Utils::CUDA::Convert::ToString(hostMax[0]);
                for (unsigned int i = 1; i < size; ++i){
                    out << ", " << Utils::CUDA::Convert::ToString(hostMax[i]);
                }
                
                return out.str();
            }
            
            std::string AABBVar::MinToString(unsigned int size) {
                std::ostringstream out;
                
                float3 hostMin[size];
                cudaMemcpy(hostMin, min, size * sizeof(float3), cudaMemcpyDeviceToHost);
                CHECK_FOR_CUDA_ERROR();
                
                out << Utils::CUDA::Convert::ToString(hostMin[0]);
                for (unsigned int i = 1; i < size; ++i){
                    out << ", " << Utils::CUDA::Convert::ToString(hostMin[i]);
                }
                
                return out.str();
            }

            
        }
    }
}
