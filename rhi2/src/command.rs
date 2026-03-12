use crate::{
    buffer::Buffer,
    graphics_pipeline::{GraphicsAttach, GraphicsPipeline},
    image::Image,
    shader::ShaderSet,
};

pub enum GraphicsCommand<B: Buffer, SS: ShaderSet> {
    BindVBs(Vec<B>),
    BindIB { ib: B, is_16bit: bool },
    BindSets(Vec<SS>),
    SetPC(Vec<u8>),
}

pub enum Command<B: Buffer, I: Image, GP: GraphicsPipeline, GA: GraphicsAttach, SS: ShaderSet> {
    CopyB2B {
        src: B,
        src_offset: usize,
        dst: B,
        dst_offset: usize,
    },
    CopyB2I {
        src: B,
        dst: I,
    },
    Graphics {
        pipeline: GP,
        attach: GA,
        commands: Vec<GraphicsCommand<B, SS>>,
    },
    Blit {
        src: I,
        dst: I,
    },
    Present(usize),
}

pub trait CmdFuture {
    fn wait();
}
