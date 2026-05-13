#version 450

layout(location = 0) in vec2 inUV;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler mySampler;
layout(set = 1, binding = 0) uniform texture2D myTexture;


void main()
{
	outColor = texture(sampler2D(myTexture, mySampler), inUV);
}