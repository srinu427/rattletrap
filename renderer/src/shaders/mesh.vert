#version 450

layout(location = 0) in vec4 inPos;
layout(location = 1) in vec4 inUV;
layout(location = 2) in vec4 inN;
layout(location = 3) in vec4 inT;
layout(location = 4) in vec4 inBT;

layout(location = 0) out vec2 outUV;

struct Cam3d {
    vec4 eye_fov;
    vec4 dir_aspect;
    vec4 up;
    mat4 proj_view;
};

layout(set = 0, binding = 0) uniform CamUni { Cam3d cam; } cam_data;
layout(set = 1, binding = 0) uniform ModelTransform { mat4 mat; } model_transform;

void main()
{
    outUV = inUV.xy;
    gl_Position = cam_data.cam.proj_view * model_transform.mat * inPos;
}