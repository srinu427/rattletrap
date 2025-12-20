#version 450

layout (location = 0) in vec4 inPos;
layout (location = 1) in vec4 inUv;

layout (location = 0) out vec4 outUv;

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
	gl_Position = camera.proj_view * inPos;
}