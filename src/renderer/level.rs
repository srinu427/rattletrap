use std::{fs, sync::LazyLock};

use anyhow::Context;
use regex::Regex;

use crate::renderer::mesh::Mesh;

static VEC3_STR: &str = "\\(([0-9.]+) +([0-9.]+) +([0-9.]+)\\)";

static VEC3_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(VEC3_STR).unwrap());

fn split_vec3(s: &str) -> Option<(glam::Vec3, &str)> {
    let cap = VEC3_RE.captures(s)?;
    let x: f32 = cap.get(1).map(|s| s.as_str().parse().ok()).flatten()?;
    let y: f32 = cap.get(2).map(|s| s.as_str().parse().ok()).flatten()?;
    let z: f32 = cap.get(3).map(|s| s.as_str().parse().ok()).flatten()?;
    let rem = &s[cap.get_match().end()..];
    Some((glam::vec3(x, y, z), rem))
}

pub fn parse_rect(tokens: &str) -> anyhow::Result<Mesh> {
    let (name, tokens) = tokens
        .split_once(char::is_whitespace)
        .context("no name given to rect")?;
    let (r_inp_type, tokens) = tokens
        .split_once(char::is_whitespace)
        .context("no rect input type specified")?;
    match r_inp_type {
        "CUV" => {
            let (c, tokens) = split_vec3(tokens).context("rect center position vec3 expected")?;
            let (u, tokens) = split_vec3(tokens).context("rect u direction vec3 expected")?;
            let (v, _tokens) = split_vec3(tokens).context("rect v direction vec3 expected")?;
            Ok(Mesh::rect_cuv(name, c, u, v))
        }
        _ => Err(anyhow::Error::msg(format!(
            "unknown rect input type: {r_inp_type}"
        ))),
    }
}

pub fn parse_geo(tokens: &str) -> anyhow::Result<Mesh> {
    let (geo_type, tokens) = tokens
        .split_once(char::is_whitespace)
        .context("can't find geo type token")?;
    match geo_type {
        "RECT" => parse_rect(tokens),
        _ => Err(anyhow::Error::msg(format!("unknown geo type: {geo_type}"))),
    }
}

pub fn parse_lvl(path: &str) -> anyhow::Result<Vec<Mesh>> {
    let mut meshes = vec![];
    let file_data = fs::read_to_string(path)?;
    for line in file_data.lines() {
        let Some((inp_type, tokens)) = line.split_once(char::is_whitespace) else {
            continue;
        };
        match inp_type {
            "GEO" => match parse_geo(tokens) {
                Ok(mesh) => meshes.push(mesh),
                Err(e) => eprintln!("{e}. skipping geo"),
            },
            _ => {
                eprintln!("unknown input type '{inp_type}'")
            }
        }
    }
    Ok(meshes)
}
