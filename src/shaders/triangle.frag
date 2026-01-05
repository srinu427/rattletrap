#version 450

layout (location = 0) in vec4 inUv;

layout (location = 0) out vec4 outFragColor;

layout(set = 1, binding = 0) uniform sampler2D tex;

void main()
{
	outFragColor = texture(tex, inUv.xy);
}