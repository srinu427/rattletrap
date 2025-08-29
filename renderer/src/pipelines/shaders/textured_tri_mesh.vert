#version 460 core

#include "common_structs.glsl"

layout (location = 0) out vec3 outPosition;
layout (location = 1) out vec2 outUV;
layout (location = 2) flat out uint objId;

layout(std430, set = 0, binding = 0) buffer readonly ssbo1 { GpuVertex vertex_buffer []; }; 
layout(std430, set = 0, binding = 1) buffer readonly ssbo2 { Camera camera;};

vec4 invert_y_axis(vec4 v) {
    return vec4(v.x, -v.y, v.z, v.w);
}

vec3 getVertexPos(int index) {
    return vec3(
        vertex_buffer[index].pos[0],
        vertex_buffer[index].pos[1],
        vertex_buffer[index].pos[2]
    );
}

vec2 getVertexUv(int index) {
    return vec2(
        vertex_buffer[index].uv[0],
        vertex_buffer[index].uv[1]
    );
}

void main() {
    vec3 inPosition = getVertexPos(gl_VertexIndex);
    vec2 inUV = getVertexUv(gl_VertexIndex);
    outPosition = inPosition;
    outUV = inUV;
    objId = vertex_buffer[gl_VertexIndex].obj_id;
    gl_Position = invert_y_axis(camera.view_proj_mat * vec4(inPosition, 1.0));
    // debugPrintfEXT("My vec is %v", gl_Position);
}