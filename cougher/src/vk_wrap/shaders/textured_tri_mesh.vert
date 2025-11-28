#version 450

#include "common_structs.glsl"

layout (location = 0) out vec4 outPosition;
layout (location = 1) out vec2 outUV;
layout (location = 2) flat out uint objId;

layout(std430, set = 0, binding = 0) buffer readonly ssbo1 { GpuVertex vertex_buffer []; };
layout(std430, set = 0, binding = 1) buffer readonly ssbo2 { uint index_buffer []; };
layout(std430, set = 0, binding = 2) buffer readonly ssbo3 { Camera camera;};

vec4 invert_y_axis(vec4 v) {
    return vec4(v.x, -v.y, v.z, v.w);
}

void main() {
    uint vert_index = index_buffer[gl_VertexIndex];
    vec4 inPosition = vertex_buffer[vert_index].pos;
    vec2 inUV = vertex_buffer[vert_index].uv;
    outPosition = inPosition;
    outUV = inUV;
    objId = vertex_buffer[vert_index].obj_id;
    gl_Position = invert_y_axis(camera.view_proj_mat * inPosition);
    // debugPrintfEXT("My vec is %v4f\n", gl_Position);
    // debugPrintfEXT("My matid is %u\n", objId);
}