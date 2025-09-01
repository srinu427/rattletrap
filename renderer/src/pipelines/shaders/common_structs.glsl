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

struct ObjectInfo {
  uint sampler_id;
  uint tex_id;
};

struct GpuVertex {
  float pos[3];
  float uv[2];
  uint obj_id;
};

struct GpuTriangle {
  float normal[3];
  float tangent[3];
  float bitangent[3];
};