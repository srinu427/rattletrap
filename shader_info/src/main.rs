use std::{env, fs};

use anyhow::Context;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    println!("{:#?}", &args);
    if args.len() != 2 {
        return Err(anyhow::Error::msg(format!(
            "expected 1 arg. found: {}",
            args.len() - 1
        )));
    }
    let shader_path = unsafe { args.get_unchecked(1) };
    println!("shader path: {shader_path}");
    let glsl_data =
        fs::read_to_string(shader_path).context(format!("IO error at {shader_path}"))?;
    let mut naga_fe = naga::front::glsl::Frontend::default();
    let options = naga::front::glsl::Options::from(naga::ShaderStage::Vertex);
    let naga_ir = naga_fe
        .parse(&options, &glsl_data)
        .context("parsing GLSL failed")?;
    println!("{:#?}", naga_ir.entry_points);
    Ok(())
}
