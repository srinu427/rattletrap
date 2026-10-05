#version 450

#define PI 3.14159265359

struct Material {
    vec4  base_color;
    vec4  specular_color;
    vec4  transmission_color;
    vec4  subsurface_color;
    vec4  coat_color;
    vec4  fuzz_color;
    vec4  emission_color;
    vec3  transmission_scatter;
    float transmission_scatter_anisotropy;
    vec3  subsurface_radius_scale;
    float subsurface_scatter_anisotropy;
    float base_weight;
    float base_metalness;
    float base_diffuse_roughness;
    float specular_weight;
    float specular_roughness;
    float specular_roughness_anisotropy;
    float specular_ior;
    float transmission_weight;
    float transmission_depth;
    float transmission_dispersion_scale;
    float transmission_dispersion_abbe_number;
    float subsurface_weight;
    float subsurface_radius;
    float coat_weight;
    float coat_roughness;
    float coat_roughness_anisotropy;
    float coat_ior;
    float coat_darkening;
    float fuzz_weight;
    float fuzz_roughness;
    float emission_luminance;
    float thin_film_weight;
    float thin_film_thickness;
    float thin_film_ior;
    float geometry_opacity;
    uint  geometry_thin_walled;
    float _pad0;
    float _pad1;
};

struct Light {
    mat4 transform;
    vec4 position;
    vec4 color;
    vec4 direction;
};

vec3 ComputeF0(float ior, vec3 baseColor, float metalness) {
    float f0_dielectric = pow((ior - 1.0) / (ior + 1.0), 2.0);
    return mix(vec3(f0_dielectric), baseColor, metalness);
}

vec3 FresnelSchlick(float VoH, vec3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - VoH, 5.0);
}

float D_GGX(float NoH, float roughness) {
    float clampedRoughness = max(roughness, 0.045); 
    float a = clampedRoughness * clampedRoughness;
    float a2 = a * a;
    float clampedNoH = min(NoH, 1.0);
    float NoH2 = clampedNoH * clampedNoH;
    float denom = (NoH2 * (a2 - 1.0) + 1.0);
    return a2 / (PI * max(denom * denom, 0.000001));
}

float V_GGX_SmithCorrelated(float NoV, float NoL, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float GGXV = NoL * sqrt(a2 + (1.0 - a2) * (NoV * NoV));
    float GGXL = NoV * sqrt(a2 + (1.0 - a2) * (NoL * NoL));
    return 0.5 / max(GGXV + GGXL, 0.00001);
}

float D_Charlie(float NoH, float roughness) {
    float invAlpha = 1.0 / max(roughness, 0.001);
    float sinH = sqrt(max(0.0, 1.0 - NoH * NoH));
    return (2.0 + invAlpha) * pow(max(sinH, 0.00001), invAlpha) / (2.0 * PI);
}

float V_Ashikhmin(float NoV, float NoL) {
    float denom = 4.0 * (NoL + NoV - NoL * NoV);
    return 1.0 / max(denom, 0.00001);
}

float DirectionalAlbedoEON(float NoX, float roughness) {
    return 1.0 - roughness * (0.4 * (1.0 - NoX) + 0.3 * roughness);
}

vec3 BRDF_EON_Diffuse(vec3 albedo, float roughness, float NoL, float NoV, float VoH) {
    float sigma = roughness;
    float sigma2 = sigma * sigma;
    float A = 1.0 - 0.5 * (sigma2 / (sigma2 + 0.33));
    float B = 0.45 * (sigma2 / (sigma2 + 0.09));
    float s = VoH - NoL * NoV;
    float t = mix(1.0, max(NoL, NoV), step(0.0, s));
    vec3 f_single = (albedo / PI) * (A + B * (s / t));
    float E_l = DirectionalAlbedoEON(NoL, roughness);
    float E_v = DirectionalAlbedoEON(NoV, roughness);
    float E_avg = 1.0 - 0.225 * roughness;
    vec3 f_multi = (albedo * albedo / PI) * ((1.0 - E_l) * (1.0 - E_v) / (1.0 - albedo * E_avg));
    return f_single + f_multi;
}

layout(set = 0, binding = 0) uniform Camera {
    mat4 tr;
    vec4 eye;
} cam;

layout(push_constant, std430) uniform PCData {
    int obj_id;
    int mat_id;
} pc;

layout(std430, set = 1, binding = 1) readonly buffer MaterialUBO {
    Material mats[];
};
layout(std430, set = 1, binding = 2) readonly buffer LightUBO {
    Light lights[];
};

layout(location = 0) in vec3 worldPos;
layout(location = 1) in vec3 worldNorm;
layout(location = 0) out vec4 outColor;

vec3 EvaluateOpenPBR_BRDF(int mat_index, vec3 N, vec3 V, vec3 L) {
    /* Evaluates the OpenPBR material model by layering a GGX clearcoat, Charlie fuzz, and a base layer (GGX specular + EON diffuse) using energy conservation. */
    #define mat mats[mat_index]

    vec3 H = normalize(V + L);

    float NoL = clamp(dot(N, L), 0.00001, 1.0);
    float NoV = clamp(dot(N, V), 0.00001, 1.0);
    float NoH = clamp(dot(N, H), 0.00001, 1.0);
    float VoH = clamp(dot(V, H), 0.00001, 1.0);

    vec3 coatF0 = ComputeF0(mat.coat_ior, vec3(1.0), 0.0);
    vec3 coatF = FresnelSchlick(VoH, coatF0);
    float coatD = D_GGX(NoH, mat.coat_roughness);
    float coatV = V_GGX_SmithCorrelated(NoV, NoL, mat.coat_roughness);
    vec3 coatSpecular = (coatD * coatV) * coatF * mat.coat_color.rgb * mat.coat_weight;
    vec3 coatEnergyAbsorption = vec3(1.0) - (coatF * mat.coat_weight);

    float fuzzD = D_Charlie(NoH, mat.fuzz_roughness);
    float fuzzV = V_Ashikhmin(NoV, NoL);
    vec3 fuzzSpecular = (fuzzD * fuzzV) * mat.fuzz_color.rgb * mat.fuzz_weight;
    vec3 fuzzEnergyAbsorption = vec3(1.0) - fuzzSpecular;

    vec3 baseF0 = ComputeF0(mat.specular_ior, mat.base_color.rgb * mat.specular_color.rgb, mat.base_metalness);
    vec3 baseF = FresnelSchlick(VoH, baseF0);
    float baseD = D_GGX(NoH, mat.specular_roughness);
    float baseV = V_GGX_SmithCorrelated(NoV, NoL, mat.specular_roughness);
    vec3 baseSpecular = (baseD * baseV) * baseF * mat.specular_weight;

    vec3 dielectricDiffuseWeight = (vec3(1.0) - baseF) * (1.0 - mat.base_metalness);

    vec3 baseAlbedo = mat.base_color.rgb * mat.base_weight;
    vec3 baseDiffuse = BRDF_EON_Diffuse(baseAlbedo, mat.base_diffuse_roughness, NoL, NoV, VoH);

    vec3 baseLayerBRDF = baseSpecular + (baseDiffuse * dielectricDiffuseWeight);
    vec3 totalBRDF = coatSpecular + (fuzzSpecular + baseLayerBRDF * fuzzEnergyAbsorption) * coatEnergyAbsorption;

    return totalBRDF * NoL;

    #undef mat 
}

vec3 AccumulateMultiLightRadiance(int mat_index, vec3 worldPos, vec3 N, vec3 V) {
    #define mat mats[mat_index] 

    vec3 accumulatedColor = vec3(0.0);
    int light_count = lights.length(); 

    for (int i = 0; i < light_count; i++) {
        Light light = lights[i];
        
        vec3 L;
        vec3 incomingRadiance = vec3(0.0);

        if (light.position.w == 0.0) {
            L = normalize(-light.direction.xyz);
            incomingRadiance = light.color.rgb * light.color.a;
        } else {
            vec3 lightVec = light.position.xyz - worldPos;
            float dist = length(lightVec);
            
            float radius = light.direction.w;
            if (dist >= radius) continue;

            L = lightVec / dist; 

            float attenuation = 1.0 / (dist * dist + 1.0);
            float factor = dist / radius;
            float falloff = clamp(1.0 - factor * factor * factor * factor, 0.0, 1.0);
            falloff = falloff * falloff;

            incomingRadiance = light.color.rgb * light.color.a * (attenuation * falloff);
        }

        if (dot(N, L) > 0.0) {
            vec3 brdf = EvaluateOpenPBR_BRDF(mat_index, N, V, L);
            accumulatedColor += brdf * incomingRadiance;
        }
    }

    accumulatedColor += mat.emission_color.rgb * mat.emission_luminance;
    accumulatedColor += vec3(0.05) * mat.base_color.xyz;

    return accumulatedColor;
    
    #undef mat
}

void main() {
    int mat_index = int(pc.mat_id);

    vec3 V = normalize(cam.eye.xyz - worldPos);
    vec3 N = normalize(worldNorm);

    vec3 color = AccumulateMultiLightRadiance(mat_index, worldPos, N, V);
    outColor = vec4(color, mats[mat_index].geometry_opacity);
}