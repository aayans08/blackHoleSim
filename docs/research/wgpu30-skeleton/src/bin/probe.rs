// Headless Metal capability probe (wgpu 30.0.2)
fn main() {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = wgpu::Backends::METAL;
    let instance = wgpu::Instance::new(desc);
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    }))
    .expect("adapter");
    println!("INFO {:#?}", adapter.get_info());
    let f = adapter.features();
    for (name, flag) in [
        ("SHADER_F16", wgpu::Features::SHADER_F16),
        ("SHADER_F64", wgpu::Features::SHADER_F64),
        ("SHADER_I16", wgpu::Features::SHADER_I16),
        ("SHADER_INT64", wgpu::Features::SHADER_INT64),
        ("FLOAT32_FILTERABLE", wgpu::Features::FLOAT32_FILTERABLE),
        ("FLOAT32_BLENDABLE", wgpu::Features::FLOAT32_BLENDABLE),
        ("TIMESTAMP_QUERY", wgpu::Features::TIMESTAMP_QUERY),
        ("TIMESTAMP_QUERY_INSIDE_ENCODERS", wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS),
        ("TIMESTAMP_QUERY_INSIDE_PASSES", wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES),
        ("TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES", wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES),
        ("TEXTURE_FORMAT_16BIT_NORM", wgpu::Features::TEXTURE_FORMAT_16BIT_NORM),
        ("RG11B10UFLOAT_RENDERABLE", wgpu::Features::RG11B10UFLOAT_RENDERABLE),
        ("BGRA8UNORM_STORAGE", wgpu::Features::BGRA8UNORM_STORAGE),
        ("TEXTURE_COMPRESSION_BC", wgpu::Features::TEXTURE_COMPRESSION_BC),
        ("TEXTURE_COMPRESSION_ASTC", wgpu::Features::TEXTURE_COMPRESSION_ASTC),
        ("TEXTURE_COMPRESSION_ETC2", wgpu::Features::TEXTURE_COMPRESSION_ETC2),
        ("SUBGROUP", wgpu::Features::SUBGROUP),
        ("TEXTURE_BINDING_ARRAY", wgpu::Features::TEXTURE_BINDING_ARRAY),
        ("IMMEDIATES", wgpu::Features::IMMEDIATES),
        ("INDIRECT_FIRST_INSTANCE", wgpu::Features::INDIRECT_FIRST_INSTANCE),
        ("EXPERIMENTAL_RAY_QUERY", wgpu::Features::EXPERIMENTAL_RAY_QUERY),
        ("SHADER_FLOAT32_ATOMIC", wgpu::Features::SHADER_FLOAT32_ATOMIC),
        ("TEXTURE_ATOMIC", wgpu::Features::TEXTURE_ATOMIC),
        ("PIPELINE_CACHE", wgpu::Features::PIPELINE_CACHE),
        ("DUAL_SOURCE_BLENDING", wgpu::Features::DUAL_SOURCE_BLENDING),
        ("CLIP_DISTANCES", wgpu::Features::CLIP_DISTANCES),
    ] {
        println!("FEATURE {name}: {}", f.contains(flag));
    }
    println!("LIMITS {:#?}", adapter.limits());
    for fmt in [
        wgpu::TextureFormat::Rgba32Float,
        wgpu::TextureFormat::Rgba16Float,
        wgpu::TextureFormat::R32Float,
        wgpu::TextureFormat::Rg11b10Ufloat,
        wgpu::TextureFormat::Rgba8Unorm,
        wgpu::TextureFormat::Bgra8Unorm,
    ] {
        println!("FORMAT {fmt:?}: {:?}", adapter.get_texture_format_features(fmt));
    }
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: None,
        required_features: adapter.features() & (wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::SHADER_F16),
        required_limits: adapter.limits(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .expect("device");
    println!("TIMESTAMP_PERIOD_NS {}", queue.get_timestamp_period());
    // Does naga accept f64 in WGSL for Metal?
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let _m = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("f64 test"),
        source: wgpu::ShaderSource::Wgsl("@compute @workgroup_size(1) fn main() { var x: f64 = 1.0lf; x = x * 2.0lf; }".into()),
    });
    println!("F64_SHADER_ERROR {:?}", pollster::block_on(scope.pop()).map(|e| e.to_string()));
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let _m = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("f16 test"),
        source: wgpu::ShaderSource::Wgsl("enable f16; @compute @workgroup_size(1) fn main() { var x: f16 = 1.0h; x = x * 2.0h; }".into()),
    });
    println!("F16_SHADER_ERROR {:?}", pollster::block_on(scope.pop()).map(|e| e.to_string()));
}
