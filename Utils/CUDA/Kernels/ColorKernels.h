// Kernels for calculating color and light in ray tracers.
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

__constant__ float3 d_lightPosition;
__constant__ float3 d_lightAmbient;
__constant__ float3 d_lightDiffuse;
__constant__ float3 d_lightSpecular;

inline __device__ __host__ float4 PhongLighting(const float4 color, const float3 normal, const float3 point, const float3 origin, 
                                                float shadow){

    float3 lightDir = normalize(FetchDeviceData(d_lightPosition) - point);
                
    // Diffuse
    float ndotl = dot(lightDir, normal) * shadow;
    float diffuse = ndotl < 0.0f ? 0.0f : ndotl;

    // Calculate specular
    float3 reflect = 2.0f * dot(normal, lightDir) * normal - lightDir;
    reflect = normalize(reflect);
    float stemp = dot(normalize(origin - point), reflect);
    stemp = stemp < 0.0f ? 0.0f : stemp;
    float specProp = 1.0f - color.w;
#ifdef __CUDA_ARCH__
    float specular = specProp * __powf(stemp, 128.0f * specProp);
#else
    float specular = specProp * powf(stemp, 128.0f * specProp);
#endif
    
    float3 light = make_float3(color) * (FetchDeviceData(d_lightAmbient) +
                                         FetchDeviceData(d_lightDiffuse) * diffuse) +
        FetchDeviceData(d_lightSpecular) * specular * 10.0f * shadow;
                
    float alpha = color.w < specular ? specular : color.w;

    return make_float4(clamp(light, 0.0f, 1.0f), alpha);
}

inline __device__ __host__ float4 Lighting(const float3 hitCoords, 
                                           float3 &origin, float3 &direction,
                                           const float4 n0, const float4 n1, const float4 n2,
                                           const uchar4 c0, float shadow = 1.0f){

    float3 point = origin + hitCoords.x * direction;

    float3 normal = (1 - hitCoords.y - hitCoords.z) * make_float3(n0) + 
                    hitCoords.y * make_float3(n1) + 
                    hitCoords.z * make_float3(n2);
    normal = normalize(normal);

    float4 color = make_float4(c0.x / 255.0f, c0.y / 255.0f, c0.z / 255.0f, c0.w / 255.0f);

    if (color.w < 1.0f - 0.00001f){
        direction -= 0.3f * 0.5f * dot(normal, normalize(origin - point)) * normal;
        //direction = 2.0f * dot(normal, -1.0f * direction) * normal + direction;
    }else{
        direction = 2.0f * dot(normal, -1.0f * direction) * normal + direction;
    }
    direction = normalize(direction);

    color.w = 1.0f - abs(1.0f - color.w);
    color = PhongLighting(color, normal, point, origin, shadow);

    // Set the origin to the intersection point and change the ray's direction.
    origin = point + 0.00001f * direction;

    return color;
}

inline __device__ __host__ float4 BlendColor(float4 orig, float4 newColor){
    float newAlpha = newColor.w;
    newColor.w = 1.0f;
                
    return orig + (1.0f - orig.w) * newAlpha * newColor;                
}
