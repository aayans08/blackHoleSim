// Display Shader - Tone mapping and presentation
// Takes compute shader output and applies HDR tone mapping

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Full-screen triangle
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0)
    );

    var uvs = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0, -1.0)
    );

    var output: VertexOutput;
    output.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    output.uv = uvs[vertex_index];
    return output;
}

@group(0) @binding(0) var render_texture: texture_2d<f32>;
@group(0) @binding(1) var render_sampler: sampler;

// ACES tone mapping (Academy Color Encoding System)
fn aces_tonemap(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;

    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Linear to sRGB gamma correction
fn linear_to_srgb(color: vec3<f32>) -> vec3<f32> {
    let cutoff = vec3<f32>(0.0031308);
    let low = color * 12.92;
    let high = 1.055 * pow(color, vec3<f32>(1.0 / 2.4)) - 0.055;

    return select(high, low, color <= cutoff);
}

// Exposure adjustment
fn apply_exposure(color: vec3<f32>, exposure: f32) -> vec3<f32> {
    return color * pow(2.0, exposure);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Sample HDR color from compute shader output
    var color = textureSample(render_texture, render_sampler, input.uv).rgb;

    // Check for NaN (color != color is true for NaN)
    if (any(color != color)) {
        return vec4<f32>(1.0, 0.0, 1.0, 1.0);  // Magenta for debugging
    }

    // Clamp extreme values before tone mapping (tighter bounds)
    color = clamp(color, vec3<f32>(0.0), vec3<f32>(5.0));

    // Neutral exposure (compute shader now handles brightness)
    let exposure = 0.0;
    color = apply_exposure(color, exposure);

    // HDR tone mapping
    color = aces_tonemap(color);

    // Gamma correction
    color = linear_to_srgb(color);

    // Output final color
    return vec4<f32>(color, 1.0);
}
