#version 450

#define PI 3.14159265359

struct Material {
    // --- 16-byte Aligned Vectors ---
    vec4  base_color;
    vec4  specular_color;
    vec4  transmission_color;
    vec4  subsurface_color;
    vec4  coat_color;
    vec4  fuzz_color;
    vec4  emission_color;

    // --- 16-byte Groups (vec3 + float scalar filler) ---
    vec3  transmission_scatter;
    float transmission_scatter_anisotropy;

    vec3  subsurface_radius_scale;
    float subsurface_scatter_anisotropy;

    // --- 4-byte Scalars ---
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
    vec4 position;  // xyz = World Position, w = Type (0 = Directional, 1 = Point)
    vec4 color;     // rgb = Light Color, a = Intensity / Power
    vec4 direction; // xyz = Light Direction (for directional lights), w = Attenuation distance (for point lights)
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// F0 calculation from IOR
vec3 ComputeF0(float ior, vec3 baseColor, float metalness) {
    float f0_dielectric = pow((ior - 1.0) / (ior + 1.0), 2.0);
    return mix(vec3(f0_dielectric), baseColor, metalness);
}

// Schlick Fresnel
vec3 FresnelSchlick(float VoH, vec3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - VoH, 5.0);
}

// GGX Normal Distribution Function (NDF)
float D_GGX(float NoH, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NoH2 = NoH * NoH;
    float denom = (NoH2 * (a2 - 1.0) + 1.0);
    return a2 / (PI * denom * denom);
}

// Height-Correlated Smith Masking-Shadowing for GGX
float V_GGX_SmithCorrelated(float NoV, float NoL, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float GGXV = NoL * sqrt(a2 + (1.0 - a2) * (NoV * NoV));
    float GGXL = NoV * sqrt(a2 + (1.0 - a2) * (NoL * NoL));
    return 0.5 / max(GGXV + GGXL, 0.00001);
}

// Charlie Distribution (Sheen / Fuzz Layer)
float D_Charlie(float NoH, float roughness) {
    float invAlpha = 1.0 / max(roughness, 0.001);
    float sinH = sqrt(max(0.0, 1.0 - NoH * NoH));
    return (2.0 + invAlpha) * pow(sinH, invAlpha) / (2.0 * PI);
}

// Ashikhmin Visibility for Sheen / Fuzz Layer
float V_Ashikhmin(float NoV, float NoL) {
    float denom = 4.0 * (NoL + NoV - NoL * NoV);
    return 1.0 / max(denom, 0.00001);
}

// Directional directional albedo polynomial approximation for GGX/EON
float DirectionalAlbedoEON(float NoX, float roughness) {
    return 1.0 - roughness * (0.4 * (1.0 - NoX) + 0.3 * roughness);
}

// Energy-Preserving Oren-Nayar (EON) Diffuse BRDF
vec3 BRDF_EON_Diffuse(vec3 albedo, float roughness, float NoL, float NoV, float VoH) {
    float sigma = roughness;
    float sigma2 = sigma * sigma;

    float A = 1.0 - 0.5 * (sigma2 / (sigma2 + 0.33));
    float B = 0.45 * (sigma2 / (sigma2 + 0.09));

    float s = VoH - NoL * NoV;
    float t = mix(1.0, max(NoL, NoV), step(0.0, s));

    // Single-scattering term
    vec3 f_single = (albedo / PI) * (A + B * (s / t));

    // Directional albedo integration approximations
    float E_l = DirectionalAlbedoEON(NoL, roughness);
    float E_v = DirectionalAlbedoEON(NoV, roughness);
    float E_avg = 1.0 - 0.225 * roughness;

    // Multi-scattering compensation
    vec3 f_multi = (albedo * albedo / PI) * ((1.0 - E_l) * (1.0 - E_v) / (1.0 - albedo * E_avg));

    return f_single + f_multi;
}

// Computes the combined BRDF factor (Specular + Diffuse + Coat + Fuzz) for one incoming direction
vec3 EvaluateOpenPBR_BRDF(Material mat, vec3 N, vec3 V, vec3 L) {
    vec3 H = normalize(V + L);

    float NoL = max(dot(N, L), 0.00001);
    float NoV = max(dot(N, V), 0.00001);
    float NoH = max(dot(N, H), 0.00001);
    float VoH = max(dot(V, H), 0.00001);

    // 1. Coat Layer
    vec3 coatF0 = ComputeF0(mat.coat_ior, vec3(1.0), 0.0);
    vec3 coatF = FresnelSchlick(VoH, coatF0);
    float coatD = D_GGX(NoH, mat.coat_roughness);
    float coatV = V_GGX_SmithCorrelated(NoV, NoL, mat.coat_roughness);
    vec3 coatSpecular = (coatD * coatV) * coatF * mat.coat_color.rgb * mat.coat_weight;
    vec3 coatEnergyAbsorption = vec3(1.0) - (coatF * mat.coat_weight);

    // 2. Fuzz Layer
    float fuzzD = D_Charlie(NoH, mat.fuzz_roughness);
    float fuzzV = V_Ashikhmin(NoV, NoL);
    vec3 fuzzSpecular = (fuzzD * fuzzV) * mat.fuzz_color.rgb * mat.fuzz_weight;
    vec3 fuzzEnergyAbsorption = vec3(1.0) - fuzzSpecular;

    // 3. Specular Layer
    vec3 baseF0 = ComputeF0(mat.specular_ior, mat.base_color.rgb * mat.specular_color.rgb, mat.base_metalness);
    vec3 baseF = FresnelSchlick(VoH, baseF0);
    float baseD = D_GGX(NoH, mat.specular_roughness);
    float baseV = V_GGX_SmithCorrelated(NoV, NoL, mat.specular_roughness);
    vec3 baseSpecular = (baseD * baseV) * baseF * mat.specular_weight;

    // Dielectric diffuse weight
    vec3 dielectricDiffuseWeight = (vec3(1.0) - baseF) * (1.0 - mat.base_metalness);

    // 4. Diffuse Layer (EON)
    vec3 baseAlbedo = mat.base_color.rgb * mat.base_weight;
    vec3 baseDiffuse = BRDF_EON_Diffuse(baseAlbedo, mat.base_diffuse_roughness, NoL, NoV, VoH);

    // Energy Conservation Composition
    vec3 baseLayerBRDF = baseSpecular + (baseDiffuse * dielectricDiffuseWeight);
    vec3 totalBRDF = coatSpecular + (fuzzSpecular + baseLayerBRDF * fuzzEnergyAbsorption) * coatEnergyAbsorption;

    // Multiply by NoL (Lambert's Cosine Law)
    return totalBRDF * NoL;
}

// Define the hardcoded LightBuffer replacement directly
const int u_LightCount = 3;

const Light u_Lights[8] = Light[8](
    // Light 0: Main Directional Sun Light
    Light(
        vec4(0.0, 0.0, 0.0, 0.0),             // pos.xyz unused, w = 0 (Directional)
        vec4(1.0, 0.95, 0.85, 4.0),           // Warm Sunlight, Intensity 4.0
        vec4(normalize(vec3(0.5, 1.0, 0.3)),  // Direction towards light
             0.0)                             // Radius unused
    ),

    // Light 1: Cool Fill Light (Directional)
    Light(
        vec4(0.0, 0.0, 0.0, 0.0),             // pos.xyz unused, w = 0 (Directional)
        vec4(0.4, 0.6, 1.0, 1.5),             // Cool Sky Fill, Intensity 1.5
        vec4(normalize(vec3(-0.5, 0.5, -0.5)),// Opposite direction
             0.0)
    ),

    // Light 2: Warm Point Light (e.g. Lamp/Torch nearby)
    Light(
        vec4(2.0, 3.0, 1.0, 1.0),             // Position (xyz), w = 1 (Point)
        vec4(1.0, 0.5, 0.1, 15.0),            // Orange/Warm Color, Intensity 15.0
        vec4(0.0, 0.0, 0.0, 8.0)              // direction.xyz unused, w = 8.0 (Radius)
    ),

    // Light 3 to 7: Dummy/Zeroed entries to fill the 8-element array
    Light(vec4(0.0), vec4(0.0), vec4(0.0)),
    Light(vec4(0.0), vec4(0.0), vec4(0.0)),
    Light(vec4(0.0), vec4(0.0), vec4(0.0)),
    Light(vec4(0.0), vec4(0.0), vec4(0.0)),
    Light(vec4(0.0), vec4(0.0), vec4(0.0))
);

vec3 AccumulateMultiLightRadiance(Material mat, vec3 worldPos, vec3 N, vec3 V) {
    vec3 accumulatedColor = vec3(0.0);

    for (int i = 0; i < u_LightCount; ++i) {
        Light light = u_Lights[i];
        
        vec3 L;
        vec3 incomingRadiance = vec3(0.0);

        if (light.position.w == 0.0) {
            // --- DIRECTIONAL LIGHT ---
            L = normalize(-light.direction.xyz);
            incomingRadiance = light.color.rgb * light.color.a;
        } else {
            // --- POINT LIGHT ---
            vec3 lightVec = light.position.xyz - worldPos;
            float dist = length(lightVec);
            
            // Early exit if out of radius range
            float radius = light.direction.w;
            if (dist >= radius) continue;

            L = lightVec / dist; // Normalize

            // Physical inverse-square falloff with smooth radius windowing
            float attenuation = 1.0 / (dist * dist + 1.0);
            float factor = dist / radius;
            float falloff = clamp(1.0 - factor * factor * factor * factor, 0.0, 1.0);
            falloff = falloff * falloff;

            incomingRadiance = light.color.rgb * light.color.a * (attenuation * falloff);
        }

        // Evaluate BRDF if surface faces light
        if (dot(N, L) > 0.0) {
            vec3 brdf = EvaluateOpenPBR_BRDF(mat, N, V, L);
            accumulatedColor += brdf * incomingRadiance;
        }
    }

    // Add Self-Emission once
    accumulatedColor += mat.emission_color.rgb * mat.emission_luminance;

    return accumulatedColor;
}

layout(set = 0, binding = 0) uniform Camera {
    mat4 tr;
    vec4 eye;
} cam;

layout(push_constant, std430) uniform PCData {
    int obj_id;
    int mat_id;
} pc;

layout(std430, set = 2, binding = 0) readonly buffer MaterialUBO {
    Material mats[];
} material_ubo;

layout(location = 0) in vec3 worldPos;
layout(location = 1) in vec3 worldNorm;

layout(location = 0) out vec4 outColor;

void main() {
    // Explicit cast fixes the Naga expression type mismatch
    int mat_index = int(pc.mat_id);
    Material mat = material_ubo.mats[mat_index];

    // View direction vector
    vec3 V = normalize(worldPos - cam.eye.xyz);
    vec3 N = normalize(worldNorm);

    vec3 color = AccumulateMultiLightRadiance(mat, worldPos, N, V);
    outColor = vec4(color, mat.geometry_opacity);
}
