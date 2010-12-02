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

inline __device__ __host__ float4 PhongLighting(const float4 color, const float3 normal, const float3 point, const float3 origin){
    
#ifdef __CUDA_ARCH__
    float3 lightDir = normalize(d_lightPosition - point);
#else
    float3 lightDir = normalize(make_float3(0.0f, 4.0f, 0.0f) - point);
#endif
                
    // Diffuse
    float ndotl = dot(lightDir, normal);
    float diffuse = ndotl < 0.0f ? 0.0f : ndotl;

    // Calculate specular
    float3 reflect = 2.0f * dot(normal, lightDir) * normal - lightDir;
    reflect = normalize(reflect);
    float stemp = dot(normalize(origin - point), reflect);
    stemp = stemp < 0.0f ? 0.0f : stemp;
    float specProp = 1.0f - color.w;
    float specular = specProp * pow(stemp, 128.0f * specProp);

#ifdef __CUDA_ARCH__
    float3 light = make_float3(color) * (d_lightAmbient +
                                         d_lightDiffuse * diffuse) +
                                         d_lightSpecular * specular * 10.0f;
#else
    const float3 lightColor = make_float3(1.0f, 0.92f, 0.8f);
    float3 light = make_float3(color) * (lightColor * 0.3f +
                                         lightColor * 0.7f * diffuse) +
        lightColor * 0.3f * specular;
#endif
                
    float alpha = color.w < specular ? specular : color.w;

    return make_float4(clamp(light, 0.0f, 1.0f), alpha);
}

inline __device__ __host__ float4 Lighting(const float3 hitCoords, 
                                           float3 &origin, float3 &direction,
                                           const float4 n0, const float4 n1, const float4 n2,
                                           const uchar4 c0){

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
    color = PhongLighting(color, normal, point, origin);

    // Set the origin to the intersection point and change the ray's direction.
    origin = point + 0.00001f * direction;

    return color;
}

inline __device__ __host__ float4 BlendColor(float4 orig, float4 newColor){
    float newAlpha = newColor.w;
    newColor.w = 1.0f;
                
    return orig + (1.0f - orig.w) * newAlpha * newColor;                
}
            
