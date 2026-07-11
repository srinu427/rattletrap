@group(0) @binding(0) var myTexture: texture_2d<f32>;
@group(0) @binding(1) var mySampler: sampler;

struct VertexInput {
    @location(0) pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) n: vec3<f32>,
    @location(3) t: vec3<f32>,
    @location(4) bt: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>};

// Vertex Shader
@vertex
fn vertex_main(vertexData: VertexInput) -> VertexOutput {

    var output: VertexOutput;
    output.position = vec4<f32>(vertexData.pos.x, vertexData.pos.y, 0.0, 1.0);
    return output;
}

// Fragment Shader
@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Output the interpolated color for the current pixel
    return textureSample(myTexture, mySampler, input.uv);
}