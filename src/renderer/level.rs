use std::{fs, sync::LazyLock};

use regex::Regex;

use crate::renderer::mesh::Mesh;

static VEC3_STR: &str = "\\(([0-9.]+) +([0-9.]+) +([0-9.]+)\\)";

static VEC3_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(VEC3_STR).unwrap());

// static GEO_RECT_CUV_RE: LazyLock<Regex> =
//     LazyLock::new(|| Regex::new("GEO RECT rectangle CUV (0 0 0) (0.2 0 0) (0 0.2 0)").unwrap());

fn parse_as_float(s: &str) -> Option<f32> {
    let out = match s.parse::<f32>() {
        Ok(fp) => fp,
        Err(_) => {
            eprintln!("parsinig {s} as f32 failed, trying i32");
            s.parse::<i32>().ok()? as _
        }
    };
    Some(out)
}

pub fn parse_lvl(path: &str) -> anyhow::Result<Vec<Mesh>> {
    let mut meshes = vec![];
    let file_data = fs::read_to_string(path)?;
    for line in file_data.lines() {
        if let Some((_, geo_line)) = line.split_once("GEO ") {
            println!("geo_line: {geo_line}");
            if let Some((_, rect_line)) = geo_line.split_once("RECT ") {
                let rect_line = rect_line.trim();
                let Some((rect_name, rect_info_line)) = rect_line.split_once(" ") else {
                    continue;
                };
                let rect_info_line = rect_info_line.trim();
                let Some((inp_type, inp_str)) = rect_info_line.split_once(" ") else {
                    continue;
                };
                let inp_type = inp_type.trim();
                let inp_str = inp_str.trim();
                if inp_type == "CUV" {
                    let vecs: Vec<_> = VEC3_RE
                        .captures_iter(inp_str)
                        .filter_map(|caps| {
                            let x = caps.get(1).map(|s| parse_as_float(s.as_str())).flatten()?;
                            let y = caps.get(2).map(|s| parse_as_float(s.as_str())).flatten()?;
                            let z = caps.get(3).map(|s| parse_as_float(s.as_str())).flatten()?;
                            Some(glam::vec3(x, y, z))
                        })
                        .collect();
                    if vecs.len() == 3 {
                        let mesh = Mesh::rect_cuv(rect_name, vecs[0], vecs[1], vecs[2]);
                        meshes.push(mesh);
                    }
                } else {
                    println!("invalid rect input type");
                }
            } else {
                println!("invalid geo type");
            }
        }
    }
    Ok(meshes)
}
