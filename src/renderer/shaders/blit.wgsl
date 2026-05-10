struct Vert {
    @location(0) pos: vec4f,
    @location(1) uv: vec4f,
    @location(2) n: vec4f,
    @location(3) t: vec4f,
    @location(4) bt: vec4f,
}

struct CamUni {
    eye: vec3f,
    fov: f32,
    dir: vec3f,
    aspect: f32,
    up: vec3f,
    padding: f32,
    proj_view: mat4x4f}

@group(0) @binding(0) var<uniform> cam_uni: CamUni;

@vertex
fn vs_main(vert: Vert) -> @builtin(position) vec4f {
    var pos = vert.pos;
    pos.z = 0.0;
    pos.w = 1.0;
    return vert.pos;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(0.0, 0.4, 1.0, 1.0);
}