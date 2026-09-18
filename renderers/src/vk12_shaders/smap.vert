#version 450

layout(location = 0) in vec3 inPosition;

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
    vec4 worldPos = mesh_info_ubo.infos[pc.obj_id].tr * vec4(inPosition, 1.0);
    gl_Position = cam.tr * worldPos;
}
