// Variables for doing a split
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Utils/CUDA/SplitVar.h>
#include <sstream>

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {

            std::string SplitVar::SideToString(unsigned int begin, unsigned int range){
                std::ostringstream out;
                
                unsigned int sides[range];
                cudaMemcpy(sides, side + begin, range * sizeof(unsigned int), cudaMemcpyDeviceToHost);
                CHECK_FOR_CUDA_ERROR();

                if (sides[0]){
                    out << "[0: left";
                }else{
                    out << "[0: right";
                }
                
                for (unsigned int i = 1; i < range; ++i){
                    if (sides[i]){
                        out << ", " << i << " left";
                    }else{
                        out << ", " << i << " right";
                    }
                }
                out << "]";
                return out.str();
            }

            std::string SplitVar::PrefixSumToString(unsigned int begin, unsigned int range){
                std::ostringstream out;
                
                unsigned int prefix[range];
                cudaMemcpy(prefix, prefixSum + begin, range * sizeof(unsigned int), cudaMemcpyDeviceToHost);
                CHECK_FOR_CUDA_ERROR();

                out << "[0: " << prefix[0];
                
                for (unsigned int i = 1; i < range; ++i){
                    out << ", " << i << ": " << prefix[i];
                }
                out << "]";
                return out.str();
            }
            
        }
    }
}
