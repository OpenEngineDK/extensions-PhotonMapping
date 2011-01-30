// CUDA Data block.
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _CUDA_DATA_BLOCK_H_
#define _CUDA_DATA_BLOCK_H_

#include <Resources/IDataBlock.h>
#include <Meta/CUDA.h>
#include <Core/Exceptions.h>
#include <Utils/CUDA/LoggerExtensions.h>

using namespace OpenEngine::Core;

namespace OpenEngine {
    namespace Resources {
        namespace CUDA {

            template <unsigned int N, class T>
            class CUDADataBlock : public IDataBlock {
            public:
                int maxSize;
                T* hostData;

            public:
                CUDADataBlock(unsigned int s = 0, T* d = NULL)
                    : IDataBlock(s, d, ARRAY, DYNAMIC) {
                    maxSize = s;
                    if (d == NULL){
                        cudaMalloc(&this->data, N * s * sizeof(T));
#if OE_SAFE
                        cudaMemset(this->data, 127, N * s * sizeof(T));
#endif
                    }
                    this->dimension = N;
                    hostData = NULL;
                }

                CUDADataBlock(IDataBlock* block){
                    throw Exception("Not implemented");
                }

                IDataBlockPtr Clone() {
                    throw Exception("Not implemented");
                }

                ~CUDADataBlock(){
                    cudaFree(this->data);
                    if (hostData)
                        delete [] hostData;
                }

                /**
                 * Get pointer to loaded data.
                 *
                 * @return T* pointer to loaded data.
                 */
                inline T* GetData(){
                    if (!hostData)
                        hostData = new T[this->size * N];
                    cudaMemcpy(hostData, this->data, this->size * N * sizeof(T), cudaMemcpyDeviceToHost);
                    return hostData;
                }
                inline T* GetDeviceData() const{
                    return (T*) this->data;
                }

                inline void* GetVoidData() { return (void*)GetData(); }

                void Resize(unsigned int i, bool dataPersistent = true){
                    if (maxSize < i){
                        if (hostData)
                            delete [] hostData;
                        
                        if (dataPersistent){
                            T *temp;
                            
                            unsigned int copySize = min(i, this->size);
                            
                            cudaMalloc(&temp, i * N *sizeof(T));
#if OE_SAFE
                            cudaMemset(temp, 127, i * N * sizeof(T));
#endif
                            cudaMemcpy(temp, this->data, copySize * N * sizeof(T), cudaMemcpyDeviceToDevice);
                            cudaFree(this->data);
                            this->data = temp;
                            CHECK_FOR_CUDA_ERROR();
                            
                        }else{
                            cudaFree(this->data);
                            cudaMalloc(&this->data, i * N *sizeof(T));
                            CHECK_FOR_CUDA_ERROR();
                        }
                        maxSize = i;
                    }
                    this->size = i;
                }

                void Extend(unsigned int i, bool dataPersistent = true){
                    Resize(i, dataPersistent);
                }

                void Unload() {throw Exception("Not implemented");}

                std::string ToString(unsigned int index, unsigned int range) {
                    std::ostringstream out;
                    out << "[";
                    T* data = GetData();
                    for (unsigned int i = 0; i < range * N; ++i){
                        if (i % N == 0) out << "[";
                        out << data[i+index];
                        if (((i+1) % N) == 0) 
                            out << "]";
                        if (i < size * N -1)
                            out << ", ";
                    }
                    out << "]";          
                    return out.str();
                }

                std::string ToString() {
                    return ToString(0, size);
                }
            };
            
        }
    }
}

#endif
