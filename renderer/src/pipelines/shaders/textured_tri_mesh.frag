#version 460 core

#include "common_structs.glsl"

layout (location = 0) in vec4 inPosition;
layout (location = 1) in vec2 inUV;
layout (location = 2) flat in uint objId;

layout (location = 0) out vec4 outFragColor;

layout(set = 1, binding = 0) uniform sampler samplers[1];
layout(set = 2, binding = 0) uniform texture2D textures[];

void main() {
    outFragColor = texture(sampler2D(textures[nonuniformEXT(objId)], samplers[0]), inUV);
    outFragColor = vec4(1.0,1.0,1.0,1.0);
}