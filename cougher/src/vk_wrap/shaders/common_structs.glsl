#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_debug_printf : enable
// #extension GL_KHR_vulkan_glsl: enable

struct Camera {
  vec4 pos;
  vec4 look_at;
  mat4 view_proj_mat;
};

struct PointLight{
  vec4 pos;
  vec4 color;
  vec4 props;
};

struct GpuVertex {
  vec4 pos;
  vec4 normal;
  vec2 uv;
  uint obj_id;
};
