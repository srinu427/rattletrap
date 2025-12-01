#version 450

#include "common_structs.glsl"

layout (location = 0) in vec4 inPosition;
layout (location = 1) in vec2 inUV;
layout (location = 2) flat in uint objId;

layout (location = 0) out vec4 outFragColor;

layout(set = 1, binding = 0) uniform sampler2D textures[];

void main() {
    // debugPrintfEXT("My matid is %u\n", objId);
    // debugPrintfEXT("My UV is %v2f\n", inUV);
    outFragColor = texture(nonuniformEXT(textures[objId]), inUV);
    // debugPrintfEXT("UV: %v2f outFragColor: %v4f\n", inUV, outFragColor);
    // outFragColor.a = 0.0;
    // outFragColor = vec4(1.0,1.0,1.0,0.0);
}
