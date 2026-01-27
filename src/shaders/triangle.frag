#version 450
#extension GL_EXT_nonuniform_qualifier : enable

layout (location = 0) in vec4 inUv;
layout (location = 1) in vec4 inN;
layout (location = 2) in vec4 inT;
layout (location = 3) in vec4 inBt;

layout (location = 0) out vec4 outFragColor;

layout(set = 1, binding = 0) uniform sampler2D tex[];

layout( push_constant ) uniform constants
{
	uint tex_id;
} pc;

void main()
{
	vec4 albedo = texture(tex[nonuniformEXT(pc.tex_id)], inUv.xy);
	outFragColor = albedo;
}