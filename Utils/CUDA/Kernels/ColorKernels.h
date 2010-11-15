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

__device__ float4 PhongLighting(float4 color, float3 normal, float3 point, float3 origin){

    float3 lightDir = normalize(d_lightPosition - point);
                
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

    float3 light = (d_lightAmbient +
                    (d_lightDiffuse * diffuse) +
                    (d_lightSpecular * specular));
                
    float alpha = color.w < specular ? specular : color.w;

    return make_float4(clamp(make_float3(color) * light, 0.0f, 1.0f), alpha);
}

__device__ float4 Lighting(int prim, float3 hitCoords, 
                           float3 &origin, float3 &direction,
                           float4 *n0s, float4 *n1s, float4 *n2s,
                           uchar4 *c0s){

    float3 point = origin + hitCoords.x * direction;

    float3 n0 = make_float3(n0s[prim]);
    float3 n1 = make_float3(n1s[prim]);
    float3 n2 = make_float3(n2s[prim]);
                
    float3 normal = (1 - hitCoords.y - hitCoords.z) * n0 + hitCoords.y * n1 + hitCoords.z * n2;
    normal = normalize(normal);
                
    uchar4 c = c0s[prim];
    float4 color = make_float4(c.x / 255.0f, c.y / 255.0f, c.z / 255.0f, c.w / 255.0f);

    if (color.w < 1.0f - 0.00001f){
        //direction -= 0.3f * 0.5f * dot(normal, normalize(origin - point)) * normal;
        direction = 2.0f * dot(normal, -1.0f * direction) * normal + direction;
        direction = normalize(direction);
    }else{
        direction = 2.0f * dot(normal, -1.0f * direction) * normal + direction;
        direction = normalize(direction);
    }

    color.w = 1.0f - abs(1.0f - color.w);
    color = PhongLighting(color, normal, point, origin);

    // Set the origin to the intersection point and change the ray's direction.
    origin = point + 0.00001f * direction;

    return color;
}

__device__ float4 BlendColor(float4 orig, float4 newColor){
    float newAlpha = newColor.w;
    newColor.w = 1.0f;
                
    return orig + (1.0f - orig.w) * newAlpha * newColor;                
}
            
