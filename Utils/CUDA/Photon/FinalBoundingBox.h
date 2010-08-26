__global__ void FinalBoundingBox1(AABBVar aabbVars, KDPhotonUpperNode upperNodes, 
                                  unsigned int iterations){

    // Optimization steps.

    // 1. Add shared memory for holding data, owner and results.

    // 2. Cut data smem consumption in half by moving the first
    //    calculation outside the loop.

    // 3. Can data be swizzled to allow faster index calculations?
    
    unsigned int i = threadIdx.x;

    for (int d = 0; d < iterations; ++d){
        unsigned int index0 = pow(2.0f, d+1) * i;
        unsigned int index1 = index0 + pow(2.0f, d);
        if (index1 < blockDim.x){
            unsigned int w0 = aabbVars.owner[index0];
            unsigned int w1 = aabbVars.owner[index1];
            if (w0 != w1){
                // Will always only be written to by one thread, since
                // owner x and x+1 only borders in one place.
                upperNodes.aabbMin[w1] = min(upperNodes.aabbMin[w1], aabbVars.min[w1]);
                upperNodes.aabbMax[w1] = max(upperNodes.aabbMax[w1], aabbVars.max[w1]);
            }else{
                aabbVars.min[index0] = min(aabbVars.min[index0], aabbVars.min[index1]);
                aabbVars.max[index0] = max(aabbVars.max[index0], aabbVars.max[index1]);
            }
        }
        __syncthreads();
    }

    // @TODO since there aren't that many threads perhaps just let
    // everyone write the result?

    if (i == 0){
        upperNodes.aabbMin[aabbVars.owner[0]] = aabbVars.min[0];
        upperNodes.aabbMax[aabbVars.owner[0]] = aabbVars.max[0];
    }
}

/**
 * Use shared memory to hold data
 */
template<unsigned int blockSize>
__global__ void FinalBoundingBox2(AABBVar aabbVars, KDPhotonUpperNode upperNodes, 
                                  unsigned int iterations, 
                                  unsigned int activeStart, unsigned int activeRange){
    // @TODO make global const
    unsigned int ownerStart = aabbVars.owner[activeStart];
    //unsigned int ownerRange = aabbVars.owner[activeStart + activeRange] - ownerStart;

    __shared__ float3 dataMin[blockSize];
    __shared__ float3 dataMax[blockSize];
    __shared__ unsigned int owner[blockSize];
    __shared__ float3 resultMin[blockSize];
    __shared__ float3 resultMax[blockSize];
    
    unsigned int i = threadIdx.x;

    // Move data to shared mem
    dataMin[i] = aabbVars.min[i];
    dataMax[i] = aabbVars.max[i];
    owner[i] = aabbVars.owner[i] - ownerStart;
    resultMin[i] = make_float3(fInfinity);
    resultMax[i] = make_float3(-1.0f * fInfinity);

    for (int d = 0; d < iterations; ++d){
        unsigned int index0 = pow(2.0f, d+1) * i;
        unsigned int index1 = index0 + pow(2.0f, d);
        if (index1 < blockDim.x){
            unsigned int w0 = owner[index0];
            unsigned int w1 = owner[index1];
            if (w0 != w1){
                // Will always only be written to by one thread, since
                // owner x and x+1 only borders in one place.
                resultMin[w1] = min(resultMin[w1], dataMin[w1]);
                resultMax[w1] = max(resultMax[w1], dataMax[w1]);
            }else{
                dataMin[index0] = min(dataMin[index0], dataMin[index1]);
                dataMax[index0] = max(dataMax[index0], dataMax[index1]);
            }
        }
        __syncthreads();
    }

    //if (i < ownerRange){
        upperNodes.aabbMin[ownerStart + i] = resultMin[i];
        upperNodes.aabbMax[ownerStart + i] = resultMax[i];
        //}
}
/**
 * Unroll the first and last iteration. This will cut data shared
 * memory in half, only half the threads will be needed. Instead of
 * copying all the data, the data will be reduced and copied into smem
 * in one go.
 */
/*
template<unsigned int blockSize>
__global__ void FinalBoundingBox3(AABBVar aabbVars, KDPhotonUpperNode upperNodes, 
                                  unsigned int iterations, 
                                  unsigned int activeStart, unsigned int activeRange, 
                                  unsigned int ownerStart, unsigned int ownerRange){

    __shared__ float3 dataMin[blockSize];
    __shared__ float3 dataMax[blockSize];
    __shared__ unsigned int owner[blockSize];
    __shared__ float3 resultMin[blockSize * 2];
    __shared__ float3 resultMax[blockSize * 2];
    
    unsigned int i = threadIdx.x;

    // Perform first reduction and Move data to shared mem
    unsigned int index0 = i;
    unsigned int index1 = i+1;
    
    if (index0 < blockDim.x){
        unsigned int w0 = aabbVars.owner[index0 + ownerStart];
        unsigned int w1 = aabbVars.owner[index1 + ownerStart];
        if (index0 != index1){
            dataMin[index0] = aabbVars.min[index0];
            dataMax[index0] = aabbVars.max[index0];
            owner[index0] = aabbVars.owner[index0];
            resultMin[]
        }else{
            dataMin[i] = min(aabbVars.min[index0], aabbVars.min[index1]);
            dataMax[i] = max(aabbVars.max[index0], aabbVars.max[index1]);
            owner[i] = aabbVars.owner[i] - ownerStart;        
        }
    }


    owner[i] = aabbVars.owner[i] - ownerStart;
    resultMin[i] = make_float3(fInfinity);
    resultMax[i] = make_float3(-1.0f * fInfinity);

    for (int d = 0; d < iterations-1; ++d){
        unsigned int index0 = pow(2.0f, d+1) * i;
        unsigned int index1 = index0 + pow(2.0f, d);
        if (index1 < blockDim.x){
            unsigned int w0 = owner[index0];
            unsigned int w1 = owner[index1];
            if (w0 != w1){
                // Will always only be written to by one thread, since
                // owner x and x+1 only borders in one place.
                resultMin[w1] = min(resultMin[w1], dataMin[w1]);
                resultMax[w1] = max(resultMax[w1], dataMax[w1]);
            }else{
                dataMin[index0] = min(dataMin[index0], dataMin[index1]);
                dataMax[index0] = max(dataMax[index0], dataMax[index1]);
            }
        }
        __syncthreads();
    }

    if (i < ownerRange){
        upperNodes.aabbMin[ownerStart + i] = resultMin[i];
        upperNodes.aabbMax[ownerStart + i] = resultMax[i];
    }
}
*/
