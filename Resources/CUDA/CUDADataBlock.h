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

using namespace OpenEngine::Core;

namespace OpenEngine {
    namespace Resources {
        namespace CUDA {

            template <unsigned int N, class T>
            class CUDADataBlock : public IDataBlock {
            public:
                T* hostData;

            public:
                CUDADataBlock(unsigned int s = 0, T* d = NULL)
                    : IDataBlock(s, d, ARRAY, DYNAMIC) {
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

                void Resize(unsigned int i, bool dataPersistent = true){
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
                    this->size = i;
                }

                void Extend(unsigned int i, bool dataPersistent = true){
                    if (size < i)
                        Resize(i, dataPersistent);
                }

                void GetElement(unsigned int i, Math::Vector<4, unsigned char> &element) {throw Exception("Not implemented");}
                void GetElement(unsigned int i, Math::Vector<2, float> &element) {throw Exception("Not implemented");}
                void GetElement(unsigned int i, Math::Vector<3, float> &element) {throw Exception("Not implemented");}
                void GetElement(unsigned int i, Math::Vector<4, float> &element) {throw Exception("Not implemented");}
                void GetElement(unsigned int i, Math::Vector<2, double> &element) {throw Exception("Not implemented");}
                void GetElement(unsigned int i, Math::Vector<3, double> &element) {throw Exception("Not implemented");}
                void GetElement(unsigned int i, Math::Vector<4, double> &element) {throw Exception("Not implemented");}
                
                void SetElement(unsigned int i, const Math::Vector<4, unsigned char> value) {throw Exception("Not implemented");}
                void SetElement(unsigned int i, const Math::Vector<2, float> value) {throw Exception("Not implemented");}
                void SetElement(unsigned int i, const Math::Vector<3, float> value) {throw Exception("Not implemented");}
                void SetElement(unsigned int i, const Math::Vector<4, float> value) {throw Exception("Not implemented");}
                void SetElement(unsigned int i, const Math::Vector<2, double> value) {throw Exception("Not implemented");}
                void SetElement(unsigned int i, const Math::Vector<3, double> value) {throw Exception("Not implemented");}
                void SetElement(unsigned int i, const Math::Vector<4, double> value) {throw Exception("Not implemented");}

                void Unload() {throw Exception("Not implemented");}

                virtual void operator+=(const Math::Vector<4, unsigned char> value) {throw Exception("Not implemented");}
                virtual void operator+=(const Math::Vector<1, unsigned int> value) {throw Exception("Not implemented");}
                virtual void operator+=(const Math::Vector<2, float> value) {throw Exception("Not implemented");}
                virtual void operator+=(const Math::Vector<3, float> value) {throw Exception("Not implemented");}
                virtual void operator+=(const Math::Vector<4, float> value) {throw Exception("Not implemented");}
                virtual void operator+=(const Math::Vector<2, double> value) {throw Exception("Not implemented");}
                virtual void operator+=(const Math::Vector<3, double> value) {throw Exception("Not implemented");}
                virtual void operator+=(const Math::Vector<4, double> value) {throw Exception("Not implemented");}
                
                virtual IDataBlockPtr operator+(const Math::Vector<4, unsigned char> value) {throw Exception("Not implemented");}
                virtual IDataBlockPtr operator+(const Math::Vector<1, unsigned int> value) {throw Exception("Not implemented");}
                virtual IDataBlockPtr operator+(const Math::Vector<2, float> value) {throw Exception("Not implemented");}
                virtual IDataBlockPtr operator+(const Math::Vector<3, float> value) {throw Exception("Not implemented");}
                virtual IDataBlockPtr operator+(const Math::Vector<4, float> value) {throw Exception("Not implemented");}
                virtual IDataBlockPtr operator+(const Math::Vector<2, double> value) {throw Exception("Not implemented");}
                virtual IDataBlockPtr operator+(const Math::Vector<3, double> value) {throw Exception("Not implemented");}
                virtual IDataBlockPtr operator+(const Math::Vector<4, double> value) {throw Exception("Not implemented");}
                
                virtual void operator-=(const Math::Vector<4, unsigned char> value) {throw Exception("Not implemented");}
                virtual void operator-=(const Math::Vector<1, unsigned int> value) {throw Exception("Not implemented");}
                virtual void operator-=(const Math::Vector<2, float> value) {throw Exception("Not implemented");}
                virtual void operator-=(const Math::Vector<3, float> value) {throw Exception("Not implemented");}
                virtual void operator-=(const Math::Vector<4, float> value) {throw Exception("Not implemented");}
                virtual void operator-=(const Math::Vector<2, double> value) {throw Exception("Not implemented");}
                virtual void operator-=(const Math::Vector<3, double> value) {throw Exception("Not implemented");}
                virtual void operator-=(const Math::Vector<4, double> value) {throw Exception("Not implemented");}
                    
                virtual IDataBlockPtr operator-(const Math::Vector<4, unsigned char> value) {throw Exception("Not implemented");}
                virtual IDataBlockPtr operator-(const Math::Vector<1, unsigned int> value) {throw Exception("Not implemented");}
                virtual IDataBlockPtr operator-(const Math::Vector<2, float> value) {throw Exception("Not implemented");}
                virtual IDataBlockPtr operator-(const Math::Vector<3, float> value) {throw Exception("Not implemented");}
                virtual IDataBlockPtr operator-(const Math::Vector<4, float> value) {throw Exception("Not implemented");}
                virtual IDataBlockPtr operator-(const Math::Vector<2, double> value) {throw Exception("Not implemented");}
                virtual IDataBlockPtr operator-(const Math::Vector<3, double> value) {throw Exception("Not implemented");}
                virtual IDataBlockPtr operator-(const Math::Vector<4, double> value) {throw Exception("Not implemented");}

                virtual void operator*=(const float s) {throw Exception("Not implemented");}
                virtual IDataBlockPtr operator*(const float s) {throw Exception("Not implemented");}
                
                virtual void operator/=(const float s) {throw Exception("Not implemented");}
                virtual IDataBlockPtr operator/(const float s) {throw Exception("Not implemented");}

                virtual void Normalize() {throw Exception("Not implemented");}

                virtual std::string ToString() {throw Exception("Not implemented");}
            };
            
        }
    }
}

#endif
