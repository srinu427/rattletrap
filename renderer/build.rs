fn main() {
    println!("cargo::rerun-if-changed=src/pipelines/shaders");

    // // Print that the build script is running
    // println!("cargo::warning=Build script is running...");

    // // Check if glslc is available
    // let glslc_check = std::process::Command::new("glslc")
    //     .arg("--version")
    //     .output();

    // match glslc_check {
    //     Ok(output) => {
    //         println!(
    //             "cargo::warning=glslc found: {}",
    //             String::from_utf8_lossy(&output.stdout)
    //         );
    //     }
    //     Err(e) => {
    //         println!("cargo::warning=glslc not found: {e}. Please install Vulkan SDK.");
    //         return;
    //     }
    // }

    // Compile vertex shader
    let vert_result = std::process::Command::new("glslc")
        .arg("src/pipelines/shaders/textured_tri_mesh.vert")
        .arg("-o")
        .arg("src/pipelines/shaders/textured_tri_mesh.vert.spv")
        .output();

    match vert_result {
        Ok(output) => {
            if !output.status.success() {
                println!("cargo::warning=Vertex shader compilation failed:");
                println!(
                    "cargo::warning=stderr: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
                panic!("Failed to compile vertex shader");
            }
            println!("cargo::warning=Vertex shader compiled successfully");
        }
        Err(e) => {
            panic!("Failed to execute glslc for vertex shader: {}", e);
        }
    }

    // Compile fragment shader
    let frag_result = std::process::Command::new("glslc")
        .arg("src/pipelines/shaders/textured_tri_mesh.frag")
        .arg("-o")
        .arg("src/pipelines/shaders/textured_tri_mesh.frag.spv")
        .output();

    match frag_result {
        Ok(output) => {
            if !output.status.success() {
                println!("cargo::warning=Fragment shader compilation failed:");
                println!(
                    "cargo::warning=stderr: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
                panic!("Failed to compile fragment shader");
            }
            println!("cargo::warning=Fragment shader compiled successfully");
        }
        Err(e) => {
            panic!("Failed to execute glslc for fragment shader: {}", e);
        }
    }

    // println!("cargo::warning=Build script completed successfully");
}
