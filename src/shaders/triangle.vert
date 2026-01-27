#version 450

layout (location = 0) in vec4 inPos;
layout (location = 1) in vec4 inUv;
layout (location = 2) in vec4 inN;
layout (location = 3) in vec4 inT;
layout (location = 4) in vec4 inBT;

layout (location = 0) out vec4 outUv;
layout (location = 1) out vec4 outN;
layout (location = 2) out vec4 outT;
layout (location = 3) out vec4 outBT;

struct Camera {
  vec4 pos_fov;
  vec4 dir_aspect;
  vec4 up;
  mat4 proj_view;
};

layout(std140, set = 0, binding = 0) uniform readonly ssbo1 { Camera camera;};

void main()
{
  outUv = inUv;
  outN = inN;
  outT = inT;
  outBT = inBT;
	gl_Position = camera.proj_view * inPos;
}