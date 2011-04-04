// Variables used when segmenting the upper nodes
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Utils/CUDA/Segments.h>
#include <Utils/CUDA/Utils.h>

#include <Meta/CUDA.h>

#include <sstream>

namespace OpenEngine {
    namespace Utils {
        namespace CUDA {
            
            Segments::Segments()
                : maxSize(0){}

            Segments::Segments(int i)
                : maxSize(i){
                
                nodeIDs = new CUDADataBlock<int>(i);
                primitiveInfo = new CUDADataBlock<int2>(i);
                aabbMin = new CUDADataBlock<point>(i);
                aabbMax = new CUDADataBlock<point>(i);
                prefixSum = new CUDADataBlock<int>(i);
            }

            __global__ void IncreaseIDs(int *nodeIDs, const int step, const int range){
                int id = blockDim.x * blockIdx.x + threadIdx.x;
                
                if (id < range){
                    int nodeID = nodeIDs[id];
                    nodeIDs[id] = nodeID + step;
                }
            }

            void Segments::IncreaseNodeIDs(const int step){
                unsigned int blocks, threads;
                Calc1DKernelDimensions(size, blocks, threads);
                IncreaseIDs<<<blocks, threads>>>(nodeIDs->GetDeviceData(), step, size);
                CHECK_FOR_CUDA_ERROR();
            }
            
            void Segments::Resize(const int i){
                nodeIDs->Resize(i);
                primitiveInfo->Resize(i);
                aabbMin->Resize(i);
                aabbMax->Resize(i);
                prefixSum->Resize(i);
                
                maxSize = i;
                size = i;
            }

            std::string Segments::ToString(int i){
                std::ostringstream out;

                if (i >= size){
                    out << "No " << i << "'th segment\n";
                }else{
                
                    int h_nodeID;
                    cudaMemcpy(&h_nodeID, nodeIDs->GetDeviceData() + i, sizeof(int), cudaMemcpyDeviceToHost);
                    CHECK_FOR_CUDA_ERROR();
                    out << "Segment " << i << " belongs to node " << h_nodeID << "\n";

                    int2 primInfo;
                    cudaMemcpy(&primInfo, primitiveInfo->GetDeviceData() + i, sizeof(int2), cudaMemcpyDeviceToHost);
                    CHECK_FOR_CUDA_ERROR();
                    out << "Ranges over " << primInfo.y << " primitives from " << primInfo.x << "\n";
                }

                return out.str();
            }

        }
    }
}
