#version 450

layout(location = 0) in vec4 inPosition;
layout(location = 1) in vec4 inUV;
layout(location = 2) in vec4 inN;
layout(location = 3) in vec4 inT;
layout(location = 4) in vec4 inBT;

layout(location = 0) out vec2 outUV;

void main()
{
    outUV = inUV.xy;
    gl_Position = vec4(inPosition.x, inPosition.y, 0.0, 1.0);
}