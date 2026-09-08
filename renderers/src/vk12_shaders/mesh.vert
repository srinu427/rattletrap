#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec3 worldPos;
layout(location = 1) out vec3 worldNorm;

layout(set = 0, binding = 0) uniform Camera {
    mat4 tr;
    vec4 eye;
} cam;

struct MeshInfo {
    mat4 tr;
};

layout(std140, set = 1, binding = 0) readonly buffer MeshInfoUBO {
    MeshInfo infos[];
} mesh_info_ubo;

layout(push_constant, std430) uniform PCData {
    int obj_id;
    int mat_id;
} pc;

void main() {
    worldPos = (mesh_info_ubo.infos[pc.obj_id].tr * vec4(inPosition, 1.0)).xyz;
    worldNorm = (mesh_info_ubo.infos[pc.obj_id].tr * vec4(inNormal, 1.0)).xyz;
    gl_Position = cam.tr * vec4(worldPos, 1.0);
}
