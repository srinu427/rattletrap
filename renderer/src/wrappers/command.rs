use ash::vk;

use crate::wrappers::{buffer::Buffer, image::Image};

pub enum Command<'a> {
    CopyBufferToBuffer {
        src: &'a Buffer,
        dst: &'a Buffer,
        regions: Vec<vk::BufferCopy>,
    },
    CopyBufferToImage {
        src: &'a Buffer,
        dst: &'a Image,
        regions: Vec<vk::BufferImageCopy>,
    },
}

pub enum RenderCommand {
    BindPipeline {
        pipeline: usize,
    },
    BindDescriptorSets {
        sets: Vec<usize>
    }
}