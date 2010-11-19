// Kernels for creating triangle upper nodes children and splitting the triangles
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

__host__ __device__ float Volume(float3 min, float3 max){
    return (max.x - min.x) * (max.y - min.y) * (max.z - min.z);
}

__host__ __device__ float MaxPlaneArea(float3 minC, float3 maxC){
    // x and y slope
    float3 v1 = make_float3(0.0f, 0.0f, maxC.z - minC.z);
    float3 v2 = make_float3(maxC.x - minC.x, maxC.y - minC.y, 0.0f);
    float area = length(cross(v1, v2));

    // z and y slope
    v1 = make_float3(0.0f, maxC.y - minC.y, 0.0f);
    v2 = make_float3(maxC.x - minC.x, 0.0f, maxC.z - minC.z);
    area = max(area, length(cross(v1, v2)));
        
    // z and x slope
    v1 = make_float3(maxC.x - minC.x, 0.0f, 0.0f);
    v2 = make_float3(0.0f, maxC.y - minC.y, maxC.z - minC.z);
    area = max(area, length(cross(v1, v2)));

    return area;
}

__device__ bool TightTriangleBB(float3 p0, float3 p1, float3 p2,
                                float3 &aabbMin, float3 &aabbMax){
    float3 tempMin = aabbMax;
    float3 tempMax = aabbMin;

    bool hit = false;
    bool lowerX = false;
    bool lowerY = false;
    bool lowerZ = false;
    bool higherX = false;
    bool higherY = false;
    bool higherZ = false;

    // p0 -> p1 intersections
    float3 dir = p1 - p0;
    float3 minInters = (aabbMin - p0) / dir;
    float3 maxInters = (aabbMax - p0) / dir;
        
    float3 minAlphas = min(minInters, maxInters);
    float3 maxAlphas = max(minInters, maxInters);

    float nearAlpha = max(minAlphas.x, max(minAlphas.y, minAlphas.z));
    float3 inter = p0 + nearAlpha * dir;

    if (aabbMin.x <= inter.x && inter.x <= aabbMax.x &&
        aabbMin.y <= inter.y && inter.y <= aabbMax.y &&
        aabbMin.z <= inter.z && inter.z <= aabbMax.z){
        tempMin = min(tempMin, inter);
        tempMax = max(tempMax, inter);
    }else{
        lowerX |= inter.x <= aabbMin.x;
        higherX |= aabbMax.x <= inter.x;
        lowerY |= inter.y <= aabbMin.y;
        higherY |= aabbMax.y <= inter.y;
        lowerZ |= inter.z <= aabbMin.z;
        higherZ |= aabbMax.z <= inter.z;
    }

    float farAlpha = min(maxAlphas.x, min(maxAlphas.y, maxAlphas.z));
    inter = p0 + farAlpha * dir;

    if (aabbMin.x <= inter.x && inter.x <= aabbMax.x &&
        aabbMin.y <= inter.y && inter.y <= aabbMax.y &&
        aabbMin.z <= inter.z && inter.z <= aabbMax.z){
        tempMin = min(tempMin, inter);
        tempMax = max(tempMax, inter);
        hit = true;
    }else{
        lowerX |= inter.x <= aabbMin.x;
        higherX |= aabbMax.x <= inter.x;
        lowerY |= inter.y <= aabbMin.y;
        higherY |= aabbMax.y <= inter.y;
        lowerZ |= inter.z <= aabbMin.z;
        higherZ |= aabbMax.z <= inter.z;
    }

    // p1 -> p2 intersections
    dir = p2 - p1;
    minInters = (aabbMin - p1) / dir;
    maxInters = (aabbMax - p1) / dir;
        
    minAlphas = min(minInters, maxInters);
    maxAlphas = max(minInters, maxInters);        
        
    nearAlpha = max(minAlphas.x, max(minAlphas.y, minAlphas.z));
    farAlpha = min(maxAlphas.x, min(maxAlphas.y, maxAlphas.z));

    nearAlpha = max(minAlphas.x, max(minAlphas.y, minAlphas.z));
    inter = p1 + nearAlpha * dir;
        
    if (aabbMin.x <= inter.x && inter.x <= aabbMax.x &&
        aabbMin.y <= inter.y && inter.y <= aabbMax.y &&
        aabbMin.z <= inter.z && inter.z <= aabbMax.z){
        tempMin = min(tempMin, inter);
        tempMax = max(tempMax, inter);
        hit = true;
    }else{
        lowerX |= inter.x <= aabbMin.x;
        higherX |= aabbMax.x <= inter.x;
        lowerY |= inter.y <= aabbMin.y;
        higherY |= aabbMax.y <= inter.y;
        lowerZ |= inter.z <= aabbMin.z;
        higherZ |= aabbMax.z <= inter.z;
    }

    farAlpha = min(maxAlphas.x, min(maxAlphas.y, maxAlphas.z));
    inter = p1 + farAlpha * dir;

    if (aabbMin.x <= inter.x && inter.x <= aabbMax.x &&
        aabbMin.y <= inter.y && inter.y <= aabbMax.y &&
        aabbMin.z <= inter.z && inter.z <= aabbMax.z){
        tempMin = min(tempMin, inter);
        tempMax = max(tempMax, inter);
        hit = true;
    }else{
        lowerX |= inter.x <= aabbMin.x;
        higherX |= aabbMax.x <= inter.x;
        lowerY |= inter.y <= aabbMin.y;
        higherY |= aabbMax.y <= inter.y;
        lowerZ |= inter.z <= aabbMin.z;
        higherZ |= aabbMax.z <= inter.z;
    }

    // p2 -> p0 intersections
    dir = p0 - p2;
    minInters = (aabbMin - p2) / dir;
    maxInters = (aabbMax - p2) / dir;

    minAlphas = min(minInters, maxInters);
    maxAlphas = max(minInters, maxInters);        
        
    nearAlpha = max(minAlphas.x, max(minAlphas.y, minAlphas.z));
    farAlpha = min(maxAlphas.x, min(maxAlphas.y, maxAlphas.z));

    nearAlpha = max(minAlphas.x, max(minAlphas.y, minAlphas.z));
    inter = p2 + nearAlpha * dir;

    if (aabbMin.x <= inter.x && inter.x <= aabbMax.x &&
        aabbMin.y <= inter.y && inter.y <= aabbMax.y &&
        aabbMin.z <= inter.z && inter.z <= aabbMax.z){
        tempMin = min(tempMin, inter);
        tempMax = max(tempMax, inter);
        hit = true;
    }else{
        lowerX |= inter.x <= aabbMin.x;
        higherX |= aabbMax.x <= inter.x;
        lowerY |= inter.y <= aabbMin.y;
        higherY |= aabbMax.y <= inter.y;
        lowerZ |= inter.z <= aabbMin.z;
        higherZ |= aabbMax.z <= inter.z;
    }

    farAlpha = min(maxAlphas.x, min(maxAlphas.y, maxAlphas.z));
    inter = p2 + farAlpha * dir;
    if (aabbMin.x <= inter.x && inter.x <= aabbMax.x &&
        aabbMin.y <= inter.y && inter.y <= aabbMax.y &&
        aabbMin.z <= inter.z && inter.z <= aabbMax.z){
        tempMin = min(tempMin, inter);
        tempMax = max(tempMax, inter);
        hit = true;
    }else{
        lowerX |= inter.x <= aabbMin.x;
        higherX |= aabbMax.x <= inter.x;
        lowerY |= inter.y <= aabbMin.y;
        higherY |= aabbMax.y <= inter.y;
        lowerZ |= inter.z <= aabbMin.z;
        higherZ |= aabbMax.z <= inter.z;
    }

    aabbMin = max(aabbMin, tempMin);
    aabbMax = min(aabbMax, tempMax);

    bool inside = lowerX && lowerY && lowerZ && higherX && higherY && higherZ;
    if (inside){
        float3 temp = aabbMin;
        aabbMin = aabbMax;
        aabbMax = temp;
    }else if (!hit){
        aabbMin = aabbMax;
    }

    return hit || inside;
}

__host__ bool TightTriangleBB(float3 p0, float3 p1, float3 p2,
                              float3 &aabbMin, float3 &aabbMax, bool hat){
    float3 tempMin = aabbMax;
    float3 tempMax = aabbMin;

    bool hit = false;
    bool lowerX = false;
    bool lowerY = false;
    bool lowerZ = false;
    bool higherX = false;
    bool higherY = false;
    bool higherZ = false;

    // p0 -> p1 intersections
    logger.info << "p0 -> p1" << logger.end;
    float3 dir = p1 - p0;
    logger.info << "dir: " << Convert::ToString(dir) << logger.end;
    float3 minInters = (aabbMin - p0) / dir;
    logger.info << "minInters: " << Convert::ToString(minInters) << logger.end;
    float3 maxInters = (aabbMax - p0) / dir;
    logger.info << "maxInters: " << Convert::ToString(maxInters) << logger.end;
        
    float3 minAlphas = min(minInters, maxInters);
    logger.info << "minAlphas: " << Convert::ToString(minAlphas) << logger.end;
    float3 maxAlphas = max(minInters, maxInters);
    logger.info << "maxAlphas: " << Convert::ToString(maxAlphas) << logger.end;

    float nearAlpha = max(minAlphas.x, max(minAlphas.y, minAlphas.z));
    float3 inter = p0 + nearAlpha * dir;
    logger.info << "inter0: " << Convert::ToString(p0) << 
        " + " << Utils::Convert::ToString(nearAlpha) << 
        " * " << Convert::ToString(dir) << 
        " = " << Convert::ToString(inter) << logger.end;

    if (aabbMin.x <= inter.x && inter.x <= aabbMax.x &&
        aabbMin.y <= inter.y && inter.y <= aabbMax.y &&
        aabbMin.z <= inter.z && inter.z <= aabbMax.z){
        logger.info << "hit" << logger.end;
        tempMin = min(tempMin, inter);
        tempMax = max(tempMax, inter);
    }else{
        lowerX |= inter.x <= aabbMin.x;
        higherX |= aabbMax.x <= inter.x;
        lowerY |= inter.y <= aabbMin.y;
        higherY |= aabbMax.y <= inter.y;
        lowerZ |= inter.z <= aabbMin.z;
        higherZ |= aabbMax.z <= inter.z;
    }

    float farAlpha = min(maxAlphas.x, min(maxAlphas.y, maxAlphas.z));
    inter = p0 + farAlpha * dir;
    logger.info << "inter1: " << Convert::ToString(p0) << 
        " + " << Utils::Convert::ToString(farAlpha) << 
        " * " << Convert::ToString(dir) << 
        " = " << Convert::ToString(inter) << logger.end;

    if (aabbMin.x <= inter.x && inter.x <= aabbMax.x &&
        aabbMin.y <= inter.y && inter.y <= aabbMax.y &&
        aabbMin.z <= inter.z && inter.z <= aabbMax.z){
        logger.info << "hit" << logger.end;
        tempMin = min(tempMin, inter);
        tempMax = max(tempMax, inter);
        hit = true;
    }else{
        lowerX |= inter.x <= aabbMin.x;
        higherX |= aabbMax.x <= inter.x;
        lowerY |= inter.y <= aabbMin.y;
        higherY |= aabbMax.y <= inter.y;
        lowerZ |= inter.z <= aabbMin.z;
        higherZ |= aabbMax.z <= inter.z;
    }

    // p1 -> p2 intersections
    logger.info << "p1 -> p2" << logger.end;
    dir = p2 - p1;
    minInters = (aabbMin - p1) / dir;
    maxInters = (aabbMax - p1) / dir;
    logger.info << "dir: " << Convert::ToString(dir) << logger.end;
    logger.info << "minInters: " << Convert::ToString(minInters) << logger.end;
    logger.info << "maxInters: " << Convert::ToString(maxInters) << logger.end;
        
    minAlphas = min(minInters, maxInters);
    maxAlphas = max(minInters, maxInters);        
    logger.info << "minAlphas: " << Convert::ToString(minAlphas) << logger.end;
    logger.info << "maxAlphas: " << Convert::ToString(maxAlphas) << logger.end;
        
    nearAlpha = max(minAlphas.x, max(minAlphas.y, minAlphas.z));
    farAlpha = min(maxAlphas.x, min(maxAlphas.y, maxAlphas.z));

    nearAlpha = max(minAlphas.x, max(minAlphas.y, minAlphas.z));
    inter = p1 + nearAlpha * dir;
    logger.info << "inter2: " << Convert::ToString(p1) << 
        " + " << Utils::Convert::ToString(nearAlpha) << 
        " * " << Convert::ToString(dir) << 
        " = " << Convert::ToString(inter) << logger.end;
        
    if (aabbMin.x <= inter.x && inter.x <= aabbMax.x &&
        aabbMin.y <= inter.y && inter.y <= aabbMax.y &&
        aabbMin.z <= inter.z && inter.z <= aabbMax.z){
        logger.info << "hit" << logger.end;
        tempMin = min(tempMin, inter);
        tempMax = max(tempMax, inter);
        hit = true;
    }else{
        lowerX |= inter.x <= aabbMin.x;
        higherX |= aabbMax.x <= inter.x;
        lowerY |= inter.y <= aabbMin.y;
        higherY |= aabbMax.y <= inter.y;
        lowerZ |= inter.z <= aabbMin.z;
        higherZ |= aabbMax.z <= inter.z;
    }

    farAlpha = min(maxAlphas.x, min(maxAlphas.y, maxAlphas.z));
    inter = p1 + farAlpha * dir;
    logger.info << "inter3: " << Convert::ToString(p1) << 
        " + " << Utils::Convert::ToString(farAlpha) << 
        " * " << Convert::ToString(dir) << 
        " = " << Convert::ToString(inter) << logger.end;

    if (aabbMin.x <= inter.x && inter.x <= aabbMax.x &&
        aabbMin.y <= inter.y && inter.y <= aabbMax.y &&
        aabbMin.z <= inter.z && inter.z <= aabbMax.z){
        logger.info << "hit" << logger.end;
        tempMin = min(tempMin, inter);
        tempMax = max(tempMax, inter);
        hit = true;
    }else{
        lowerX |= inter.x <= aabbMin.x;
        higherX |= aabbMax.x <= inter.x;
        lowerY |= inter.y <= aabbMin.y;
        higherY |= aabbMax.y <= inter.y;
        lowerZ |= inter.z <= aabbMin.z;
        higherZ |= aabbMax.z <= inter.z;
    }

    // p2 -> p0 intersections
    logger.info << "p2 -> p0" << logger.end;
    dir = p0 - p2;
    minInters = (aabbMin - p2) / dir;
    maxInters = (aabbMax - p2) / dir;
    logger.info << "dir: " << Convert::ToString(dir) << logger.end;
    logger.info << "minInters: " << Convert::ToString(minInters) << logger.end;
    logger.info << "maxInters: " << Convert::ToString(maxInters) << logger.end;

    minAlphas = min(minInters, maxInters);
    maxAlphas = max(minInters, maxInters);        
    logger.info << "minAlphas: " << Convert::ToString(minAlphas) << logger.end;
    logger.info << "maxAlphas: " << Convert::ToString(maxAlphas) << logger.end;
        
    nearAlpha = max(minAlphas.x, max(minAlphas.y, minAlphas.z));
    farAlpha = min(maxAlphas.x, min(maxAlphas.y, maxAlphas.z));

    nearAlpha = max(minAlphas.x, max(minAlphas.y, minAlphas.z));
    inter = p2 + nearAlpha * dir;
    logger.info << "inter4: " << Convert::ToString(p2) << 
        " + " << Utils::Convert::ToString(nearAlpha) << 
        " * " << Convert::ToString(dir) << 
        " = " << Convert::ToString(inter) << logger.end;

    if (aabbMin.x <= inter.x && inter.x <= aabbMax.x &&
        aabbMin.y <= inter.y && inter.y <= aabbMax.y &&
        aabbMin.z <= inter.z && inter.z <= aabbMax.z){
        logger.info << "hit" << logger.end;
        tempMin = min(tempMin, inter);
        tempMax = max(tempMax, inter);
        hit = true;
    }else{
        lowerX |= inter.x <= aabbMin.x;
        higherX |= aabbMax.x <= inter.x;
        lowerY |= inter.y <= aabbMin.y;
        higherY |= aabbMax.y <= inter.y;
        lowerZ |= inter.z <= aabbMin.z;
        higherZ |= aabbMax.z <= inter.z;
    }

    farAlpha = min(maxAlphas.x, min(maxAlphas.y, maxAlphas.z));
    inter = p2 + farAlpha * dir;
    logger.info << "inter5: " << Convert::ToString(p2) << 
        " + " << Utils::Convert::ToString(farAlpha) << 
        " * " << Convert::ToString(dir) << 
        " = " << Convert::ToString(inter) << logger.end;
    if (aabbMin.x <= inter.x && inter.x <= aabbMax.x &&
        aabbMin.y <= inter.y && inter.y <= aabbMax.y &&
        aabbMin.z <= inter.z && inter.z <= aabbMax.z){
        logger.info << "hit" << logger.end;
        tempMin = min(tempMin, inter);
        tempMax = max(tempMax, inter);
        hit = true;
    }else{
        lowerX |= inter.x <= aabbMin.x;
        higherX |= aabbMax.x <= inter.x;
        lowerY |= inter.y <= aabbMin.y;
        higherY |= aabbMax.y <= inter.y;
        lowerZ |= inter.z <= aabbMin.z;
        higherZ |= aabbMax.z <= inter.z;
    }

    aabbMin = max(aabbMin, tempMin);
    aabbMax = min(aabbMax, tempMax);

    bool inside = lowerX && lowerY && lowerZ && higherX && higherY && higherZ;
    if (inside){
        float3 temp = aabbMin;
        aabbMin = aabbMax;
        aabbMax = temp;
    }else if (!hit){
        aabbMin = aabbMax;
    }

    return hit || inside;
}

__global__ void AdjustBoundingBox(float4* aabbMins, float4* aabbMaxs,
                                  float4* p0s, float4* p1s, float4* p2s,
                                  float4 *origMins, float4 *origMaxs,
                                  float *origArea,
                                  int primitives){

    const int id = blockDim.x * blockIdx.x + threadIdx.x;
        
    if (id < primitives){
        float4 hat = aabbMins[id];
        const int primID = hat.w;
        float3 minCorner = make_float3(hat);
        float3 maxCorner = make_float3(aabbMaxs[id]);

        float3 ps[3];
        ps[0] = make_float3(p0s[primID]);
        ps[1] = make_float3(p1s[primID]);
        ps[2] = make_float3(p2s[primID]);

        bool hit = TightTriangleBB(ps[0], ps[1], ps[2], minCorner, maxCorner);
                
        float3 origMin = make_float3(origMins[primID]);
        float3 origMax = make_float3(origMaxs[primID]);
            
        float origMaxArea = MaxPlaneArea(origMin, origMax);
        float newMaxArea = MaxPlaneArea(minCorner, maxCorner);

        float newArea = sqrt(newMaxArea / origMaxArea) * origArea[primID];
        newArea = hit ? min(newMaxArea, newArea) : 0.0f;

        aabbMins[id] = make_float4(minCorner, primID);
        aabbMaxs[id] = make_float4(maxCorner, newArea);
    }
}
