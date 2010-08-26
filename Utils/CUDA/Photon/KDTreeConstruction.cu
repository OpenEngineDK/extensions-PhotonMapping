// KD tree structs for CUDA
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Math/RandomGenerator.h>
#include <Meta/CUDA.h>
#include <Resources/IDataBlock.h>
#include <Utils/CUDA/Convert.h>
#include <Logging/Logger.h>

#include <Utils/CUDA/AABBVar.h>
#include <Utils/CUDA/SplitVar.h>
#include <Utils/CUDA/Photon/KDPhotonUpperNode.h>
#include <Utils/CUDA/KDTree.h>
#include <Utils/CUDA/Utils.h>

#include <algorithm>

using namespace OpenEngine::Resources;
using namespace OpenEngine::Utils::CUDA;
using namespace OpenEngine::Utils::CUDA::Photon;

photon photons;
KDPhotonUpperNode upperNodes;
AABBVar aabbVars;
SplitVar splitVars;
unsigned int timerID;

/**
 * Initialize photons to default values.
 */
__global__ void InitDevicePhotons(photon photons, unsigned int amount){
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    unsigned int gridSize = blockDim.x * 2 * gridDim.x;

    while (i < amount){
        photons.pos[i] = make_float3(i, amount - i, 1.025f);
        //photons.assoc[i] = i;
        if (i + blockDim.x < amount){
            photons.pos[i + blockDim.x] = make_float3(i + blockDim.x, amount - i - blockDim.x, 1.023f);
            //photons.assoc[i + blockDim.x] = i + blockDim.x;
        }
        i += gridSize;
    }
}

/**
 * Allocated space for photons and their acceleration structures on the GPU.
 */
void InitPhotons(unsigned int amount) {
    logger.info << "Initialize " << amount << " photons" << logger.end;

    // Create timer
    cutCreateTimer(&timerID);

    // AABB calc vars
    aabbVars = AABBVar(MAX_BLOCKS);

    // Allocate photons on GPU
    photons.maxSize = photons.size = amount;
    cudaMalloc(&(photons.pos), amount * sizeof(float3));
    //cudaMalloc(&(photons.assoc), amount * sizeof(unsigned int));
    CHECK_FOR_CUDA_ERROR();

    // Initialize the photons
    InitDevicePhotons<<< 64, 256>>>(photons, amount);
    CHECK_FOR_CUDA_ERROR();

    float3 hat[amount];
    Math::RandomGenerator rand;
    for (unsigned int i = 0; i < amount / 2; ++i)
        hat[i] = make_float3(rand.UniformFloat(0.0f, 5.0f),
                             rand.UniformFloat(0.0f, 10.0f),
                             rand.UniformFloat(0.0f, 10.0f));
    for (unsigned int i = amount / 2; i < amount; ++i)
        hat[i] = make_float3(rand.UniformFloat(5.0f, 10.0f),
                             rand.UniformFloat(0.0f, 10.0f),
                             rand.UniformFloat(0.0f, 10.0f));
    
    cudaMemcpy(photons.pos, hat, amount * sizeof(float3), cudaMemcpyHostToDevice);

    // Calculate amount of nodes probably required
    unsigned int upperNodeSize = 2.5f * photons.maxSize / BUCKET_SIZE;    
    upperNodes.Init(upperNodeSize);

    splitVars.Init(amount);

}

/**
 * Computes the axis aligned bounding box for the given photon node.
 *
 * Could be optimized to work for several nodes to give increased
 * performance? Pass a list of photon KD nodes and let each thread
 * compute it's bounds?
 *
 * Send precalculated start- and endIndex? Saves every thread to
 * lookup the same value but is it faster?
 *
 * Based on NVIDIA's reduction sample.
 */
template <class T, unsigned int blockSize> 
__global__ void CalcBoundingBox(photon photons, 
                                KDPhotonUpperNode upperNodes,
                                AABBVar aabbVars,
                                unsigned int nodeID,
                                unsigned int blockOffset) {
    // now that we are using warp-synchronous programming (below)
    // we need to declare our shared memory volatile so that the compiler
    // doesn't reorder stores to it and induce incorrect behavior.
    /*volatile*/ T* sdata = SharedMemory<T>();

    unsigned int tid = threadIdx.x;
    unsigned int i = upperNodes.startIndex[nodeID] + blockIdx.x*blockSize + threadIdx.x;
    unsigned int gridSize = blockSize*gridDim.x;

    // Do first reduction outside the loop to avoid assigning dummy values.
    T localMax, localMin;
    if (false){
        localMax = localMin = photons.pos[i];
        i += gridSize;
    }else{
        localMax = make_float3(-1.0 * fInfinity);
        localMin = make_float3(fInfinity);
    }

    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < upperNodes.startIndex[nodeID] + upperNodes.range[nodeID])
    {         
        localMax = max(localMax, photons.pos[i]);
        localMin = min(localMin, photons.pos[i]);
        i += gridSize;
    } 

    // each thread puts its local sum into shared memory 
    sdata[tid] = localMax;
    sdata[tid + blockSize] = localMin;
    __syncthreads();
    
    // Do reduction in shared memory.

    // Compiler can't unroll loop, so do this manually for better
    // performance?
    // Haven't now because this is easier for development.
    for (unsigned int i = blockSize; i > 32; i /= 2){
        unsigned int offset = i / 2;
        if (tid < offset){
            sdata[tid] = localMax = max(localMax, sdata[tid + offset]);
            sdata[tid + blockSize] = localMin = min(localMin, sdata[tid + offset + blockSize]);
        }
        __syncthreads();
    }

    for (unsigned int i = min(32, blockSize); i > 1; i /= 2){
        unsigned int offset = i / 2;
        sdata[tid] = localMax = max(localMax, sdata[tid + offset]);
        sdata[tid + blockSize] = localMin = min(localMin, sdata[tid + offset + blockSize]);
        __syncthreads(); 
    }

    // write result for this block to global mem 
    if (tid == 0) {
        aabbVars.max[blockIdx.x + blockOffset] = sdata[0];
        aabbVars.min[blockIdx.x + blockOffset] = sdata[blockSize];
        aabbVars.owner[blockIdx.x + blockOffset] = nodeID;
    }
}

#include <Utils/CUDA/Photon/FinalBoundingBox.h>

void ComputeBoundingBox(unsigned int activeIndex, unsigned int activeRange){
    unsigned int nodeRanges[activeRange];
    cudaMemcpy(nodeRanges, upperNodes.range + activeIndex, 
               activeRange * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    unsigned int nodeID = activeIndex;
    while (nodeID < activeIndex + activeRange){
        unsigned int blocksUsed = 0;
        do {
            /*
             * @TODO Extend to compute multiple bounding boxes instead of
             * computing one at a time.
             */

            logger.info << "AABB for node " << nodeID << logger.end;
            
            unsigned int size = nodeRanges[nodeID - activeIndex];
        
            int threads = (size < MAX_THREADS * 2) ? NextPow2((size + 1)/ 2) : MAX_THREADS;
            int blocks = (size + (threads * 2 - 1)) / (threads * 2);
            blocks = min(MAX_BLOCKS, blocks);
            int smemSize = (threads <= 32) ? 4 * threads * sizeof(float3) : 2 * threads * sizeof(float3);
            
            logger.info << "Threads " << threads << logger.end;
            logger.info << "blocks " << blocks << logger.end;

            // Execute kernel
            switch(threads){
            case 512:
                CalcBoundingBox<float3, 512><<< blocks, threads, smemSize >>>(photons, upperNodes, aabbVars, nodeID, blocksUsed);
                break;
            case 256:
                CalcBoundingBox<float3, 256><<< blocks, threads, smemSize >>>(photons, upperNodes, aabbVars, nodeID, blocksUsed);
            break;
            case 128:
                CalcBoundingBox<float3, 128><<< blocks, threads, smemSize >>>(photons, upperNodes, aabbVars, nodeID, blocksUsed);
                break;
            case 64:
                CalcBoundingBox<float3, 64><<< blocks, threads, smemSize >>>(photons, upperNodes, aabbVars, nodeID, blocksUsed);
                break;
            case 32:
                CalcBoundingBox<float3, 32><<< blocks, threads, smemSize >>>(photons, upperNodes, aabbVars, nodeID, blocksUsed);
                break;
            case 16:
                CalcBoundingBox<float3, 16><<< blocks, threads, smemSize >>>(photons, upperNodes, aabbVars, nodeID, blocksUsed);
                break;
            case 8:
                CalcBoundingBox<float3, 8><<< blocks, threads, smemSize >>>(photons, upperNodes, aabbVars, nodeID, blocksUsed);
                break;
            case 4:
                CalcBoundingBox<float3, 4><<< blocks, threads, smemSize >>>(photons, upperNodes, aabbVars, nodeID, blocksUsed);
            break;
            case 2:
                CalcBoundingBox<float3, 2><<< blocks, threads, smemSize >>>(photons, upperNodes, aabbVars, nodeID, blocksUsed);
                break;
            }
            CHECK_FOR_CUDA_ERROR();
            
            blocksUsed += blocks;
            nodeID++;
        }while(blocksUsed < MAX_BLOCKS && nodeID < activeIndex + activeRange);

        // Run segmented reduce ie FinalBoundingBox
        unsigned int iterations = log2(blocksUsed * 1.0f);
        //cutResetTimer(timerID);
        //cutStartTimer(timerID);
        FinalBoundingBox1<<<1, blocksUsed>>>(aabbVars, upperNodes, iterations);
        CHECK_FOR_CUDA_ERROR();
        /*
          cudaThreadSynchronize();
          CHECK_FOR_CUDA_ERROR();
          cutStopTimer(timerID);
          
          logger.info << "Final bounding box time: " << cutGetTimerValue(timerID) << "ms" << logger.end;
        */    
    }

}

/**
 * Setup splitting planes and link to children.
 */
__global__ void UpperNodeSplitPos(KDPhotonUpperNode upperNodes,
                                  unsigned int activeIndex){
    
    unsigned int nodeID = threadIdx.x + activeIndex;
    
    // the nodes aabb size
    float3 bbSize = upperNodes.aabbMax[nodeID] - upperNodes.aabbMin[nodeID];
    float3 median = bbSize * 0.5 + upperNodes.aabbMin[nodeID];

    // Calculate splitting plane
    bool yAboveX = bbSize.x < bbSize.y;
    float max = yAboveX ? bbSize.y : bbSize.x;
    float split = yAboveX ? median.y : median.x;
    char axis = yAboveX ? KDPhotonUpperNode::Y : KDPhotonUpperNode::X;
    bool zHigher = max < bbSize.z;
    upperNodes.splitPos[nodeID] = zHigher ? median.z : split;
    upperNodes.info[nodeID] = zHigher ? KDPhotonUpperNode::Z : axis;

    // Set link to parent nodes from child nodes
    unsigned int leftChild = upperNodes.child[nodeID] = upperNodes.size + 2 * nodeID;
    upperNodes.parent[leftChild] = upperNodes.parent[leftChild+1] = nodeID;
}

template<char splitPlane>
__global__ void CalcSplitSide(SplitVar splitVars, 
                  KDPhotonUpperNode upperNodes,
                  photon photons, unsigned int photonRange,
                  unsigned int nodeID){
    unsigned int posStart = upperNodes.startIndex[nodeID];
    float splitPos = upperNodes.splitPos[nodeID];
      
    unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int stepSize = gridDim.x * blockDim.x;

    while(id < photonRange){
        unsigned int index = id + posStart;
        if (splitPlane == KDPhotonUpperNode::X)
            splitVars.side[index] = photons.pos[index].x < splitPos;
        else if (splitPlane == KDPhotonUpperNode::Y)
            splitVars.side[index] = photons.pos[index].y < splitPos;
        else if (splitPlane == KDPhotonUpperNode::Z)
            splitVars.side[index] = photons.pos[index].z < splitPos;
        id += stepSize;
    }
}

void SplitUpperNodes(unsigned int activeIndex, unsigned int activeRange){
    
    UpperNodeSplitPos<<< 1, activeRange>>>(upperNodes, activeIndex);

    // Split each node along the spatial median
    for (unsigned int nodeID = activeIndex; nodeID < activeIndex + activeRange; ++nodeID){
        char axis;
        cudaMemcpy(&axis, upperNodes.info+nodeID, sizeof(char), cudaMemcpyDeviceToHost);
        unsigned int photonRange;
        cudaMemcpy(&photonRange, upperNodes.range+nodeID, sizeof(unsigned int), cudaMemcpyDeviceToHost);

        unsigned int blocks, threads;
        Calc1DKernelDimensions(photonRange, blocks, threads);
        
        switch (axis) {
        case KDPhotonUpperNode::X:
            CalcSplitSide<KDPhotonUpperNode::X><<<blocks, threads>>>(splitVars, upperNodes, photons, photonRange, nodeID);
        case KDPhotonUpperNode::Y:
            CalcSplitSide<KDPhotonUpperNode::Y><<<blocks, threads>>>(splitVars, upperNodes, photons, photonRange, nodeID);
        case KDPhotonUpperNode::Z:
            CalcSplitSide<KDPhotonUpperNode::Z><<<blocks, threads>>>(splitVars, upperNodes, photons, photonRange, nodeID);
        }
        CHECK_FOR_CUDA_ERROR();

    }
}

void CreateUpperPhotonNodes(unsigned int activeIndex, unsigned int activeRange, 
                            unsigned int &childrenCreated, unsigned int &lowerCreated){

    
    ComputeBoundingBox(activeIndex, activeRange);
    CHECK_FOR_CUDA_ERROR();

    SplitUpperNodes(activeIndex, activeRange);

    childrenCreated = 0;
    lowerCreated = 0;
}

/**
 * Create the KD tree for photons. The photons are already on the GPU.
 */
void MapPhotons(){
    unsigned int childrenCreated = 1, lowerCreated;

    // Initialize kd root node data
    unsigned int i = 0;
    cudaMemcpy(upperNodes.startIndex, &i, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(upperNodes.range, &(photons.size), sizeof(unsigned int), cudaMemcpyHostToDevice);
    upperNodes.size = 1;

    // Create upper nodes
    while (childrenCreated != 0){
        CreateUpperPhotonNodes(0, 1, childrenCreated, lowerCreated);
    }

    // Print photons
    logger.info << upperNodes.ToString(0) << logger.end;


    // Create lower nodes
    
}

/**
 * Map photons to datablocks for visualization.
 */
void MapPhotonsToOpenGL(IDataBlock* pos){
    cudaGraphicsResource* resource;
    cudaGraphicsGLRegisterBuffer(&resource, pos->GetID(), cudaGraphicsMapFlagsWriteDiscard);
    CHECK_FOR_CUDA_ERROR();

    cudaGraphicsMapResources(1, &resource, 0);
    CHECK_FOR_CUDA_ERROR();
    
    float3* verts;
    size_t bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&verts, &bytes,
                                         resource);
    CHECK_FOR_CUDA_ERROR();

    cudaMemcpy(verts, photons.pos, bytes, cudaMemcpyDeviceToDevice);
    CHECK_FOR_CUDA_ERROR();

    cudaGraphicsUnmapResources(1, &resource, 0);
    CHECK_FOR_CUDA_ERROR();

    cudaGraphicsUnregisterResource(resource);
    CHECK_FOR_CUDA_ERROR();
}
