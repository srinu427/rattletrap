fn compile_shader(path: &str) {
    // Compile shader
    let comp_result = std::process::Command::new("glslc")
        .arg(path)
        .arg("-o")
        .arg(format!("{path}.spv"))
        .output();
    match comp_result {
        Ok(output) => {
            if !output.status.success() {
                println!("cargo::warning=shader compilation failed");
                println!(
                    "cargo::warning=stderr: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
                panic!("Failed to compile shader {path}");
            }
            println!("cargo::warning=Vertex shader compiled successfully");
        }
        Err(e) => {
            panic!("Failed to execute glslc for shader {path}: {e}");
        }
    }
}

fn main() {
    println!("cargo::rerun-if-changed=src/vk12/shaders");
    let shader_list = [
        "src/vk_wrap/shaders/textured_tri_mesh.vert",
        "src/vk_wrap/shaders/textured_tri_mesh.frag",
    ];

    // Start compilation
    for shader in shader_list {
        compile_shader(shader);
    }

    // println!("cargo::warning=Build script completed successfully");
}
