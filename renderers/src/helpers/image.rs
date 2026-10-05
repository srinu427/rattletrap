use std::ops::{AddAssign, Range};

use ash::vk;
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme},
};
use hashbrown::HashMap;

use crate::helpers::device::GpuCtx;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct GpuImageViewInfo {
    pub type_: vk::ImageViewType,
    pub layer_range: Range<u32>,
    pub level_range: Range<u32>,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct GpuImageAccess {
    pub layout: vk::ImageLayout,
    pub access: vk::AccessFlags,
    pub stage: vk::PipelineStageFlags,
}

impl Default for GpuImageAccess {
    fn default() -> Self {
        Self {
            layout: vk::ImageLayout::UNDEFINED,
            access: vk::AccessFlags::empty(),
            stage: vk::PipelineStageFlags::TOP_OF_PIPE,
        }
    }
}

pub fn is_depth_stencil(fmt: vk::Format) -> (bool, bool) {
    match fmt {
        vk::Format::D16_UNORM | vk::Format::D32_SFLOAT | vk::Format::X8_D24_UNORM_PACK32 => {
            (true, false)
        }
        vk::Format::D16_UNORM_S8_UINT
        | vk::Format::D24_UNORM_S8_UINT
        | vk::Format::D32_SFLOAT_S8_UINT => (true, true),
        _ => (false, false),
    }
}

pub fn image_aspect_mask(fmt: vk::Format) -> vk::ImageAspectFlags {
    let (is_d, has_s) = is_depth_stencil(fmt);
    if is_d {
        if has_s {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        } else {
            vk::ImageAspectFlags::DEPTH
        }
    } else {
        vk::ImageAspectFlags::COLOR
    }
}

pub struct GpuImage {
    pub handle: vk::Image,
    pub type_: vk::ImageType,
    pub format: vk::Format,
    pub res: (u32, u32, u32),
    pub levels: u32,
    pub allocation: Option<Allocation>,
    pub views: HashMap<GpuImageViewInfo, vk::ImageView>,
    pub access: GpuImageAccess,
}

impl GpuImage {
    pub fn new(
        ctx: &mut GpuCtx,
        type_: vk::ImageType,
        format: vk::Format,
        res: (u32, u32, u32),
        levels: u32,
        usage: vk::ImageUsageFlags,
    ) -> anyhow::Result<Self> {
        let handle = unsafe {
            ctx.device.create_image(
                &vk::ImageCreateInfo::default()
                    .array_layers(match type_ {
                        vk::ImageType::TYPE_3D => 1,
                        _ => res.2,
                    })
                    .extent(vk::Extent3D {
                        width: res.0,
                        height: res.1,
                        depth: match type_ {
                            vk::ImageType::TYPE_3D => res.2,
                            _ => 1,
                        },
                    })
                    .format(format)
                    .image_type(type_)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .mip_levels(levels)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .usage(usage),
                None,
            )?
        };
        let requirements = unsafe { ctx.device.get_image_memory_requirements(handle) };
        let allocation = ctx.allocator.allocate(&AllocationCreateDesc {
            name: &format!("image_{:?}", handle),
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })?;
        unsafe {
            ctx.device
                .bind_image_memory(handle, allocation.memory(), allocation.offset())?;
        }
        Ok(GpuImage {
            handle,
            type_,
            format,
            res,
            levels,
            allocation: Some(allocation),
            views: Default::default(),
            access: Default::default(),
        })
    }

    pub fn get_view(
        &mut self,
        ctx: &GpuCtx,
        info: GpuImageViewInfo,
    ) -> anyhow::Result<vk::ImageView> {
        let iv = self.views.get(&info).cloned();
        let iv = match iv {
            Some(t) => t,
            None => {
                let iv = unsafe {
                    ctx.device.create_image_view(
                        &vk::ImageViewCreateInfo::default()
                            .components(vk::ComponentMapping::default())
                            .format(self.format)
                            .image(self.handle)
                            .subresource_range(
                                vk::ImageSubresourceRange::default()
                                    .aspect_mask(image_aspect_mask(self.format))
                                    .base_array_layer(info.layer_range.start)
                                    .base_mip_level(info.level_range.start)
                                    .layer_count(info.layer_range.end - info.layer_range.start)
                                    .level_count(info.layer_range.end - info.layer_range.start),
                            )
                            .view_type(info.type_),
                        None,
                    )?
                };
                self.views.insert(info, iv);
                iv
            }
        };
        Ok(iv)
    }

    pub fn transition(&mut self, ctx: &GpuCtx, cb: vk::CommandBuffer, new_access: GpuImageAccess) {
        if self.access == new_access {
            return;
        }
        unsafe {
            ctx.device.cmd_pipeline_barrier(
                cb,
                self.access.stage,
                new_access.stage,
                vk::DependencyFlags::BY_REGION,
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .dst_access_mask(new_access.access)
                    .dst_queue_family_index(ctx.gpu_info.graphics_qf)
                    .image(self.handle)
                    .new_layout(new_access.layout)
                    .old_layout(self.access.layout)
                    .src_access_mask(self.access.access)
                    .src_queue_family_index(ctx.gpu_info.graphics_qf)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(image_aspect_mask(self.format))
                            .layer_count(self.res.2)
                            .level_count(self.levels),
                    )],
            );
        }
    }

    pub fn destroy(&mut self, ctx: &mut GpuCtx) {
        unsafe {
            for (_, view) in self.views.drain() {
                ctx.device.destroy_image_view(view, None);
            }
        }
        if let Some(altn) = self.allocation.take() {
            unsafe {
                ctx.device.destroy_image(self.handle, None);
            }
            if let Err(e) = ctx.allocator.free(altn) {
                log::warn!("freeing memory of buffer {:?} failed: {e}", self.handle)
            };
        }
    }
}

fn get_pool_sizes(
    bindings: &[(vk::DescriptorType, u32)],
    sets: u32,
) -> Vec<vk::DescriptorPoolSize> {
    let mut counts = HashMap::new();
    for (t, count) in bindings {
        counts.entry(*t).or_insert(0).add_assign(*count * sets);
    }
    counts
        .iter()
        .map(|(t, c)| {
            vk::DescriptorPoolSize::default()
                .ty(*t)
                .descriptor_count(*c)
        })
        .collect()
}

pub struct GpuDsl {
    pub handle: vk::DescriptorSetLayout,
    pub bindings: Vec<(vk::DescriptorType, u32)>,
    pub pools: Vec<vk::DescriptorPool>,
    pub sets: Vec<vk::DescriptorSet>,
}

impl GpuDsl {
    pub fn new(ctx: &mut GpuCtx, bindings: Vec<(vk::DescriptorType, u32)>) -> anyhow::Result<Self> {
        let dsl = unsafe {
            ctx.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(
                    &bindings
                        .iter()
                        .enumerate()
                        .map(|(i, b)| {
                            vk::DescriptorSetLayoutBinding::default()
                                .binding(i as _)
                                .descriptor_count(b.1)
                                .descriptor_type(b.0)
                                .stage_flags(vk::ShaderStageFlags::ALL)
                        })
                        .collect::<Vec<_>>(),
                ),
                None,
            )?
        };
        Ok(Self {
            handle: dsl,
            bindings,
            pools: Default::default(),
            sets: Default::default(),
        })
    }

    pub fn get_set(&mut self, ctx: &GpuCtx) -> anyhow::Result<vk::DescriptorSet> {
        let set = self.sets.pop();
        let set = match set {
            Some(t) => t,
            None => {
                let pool = unsafe {
                    ctx.device.create_descriptor_pool(
                        &vk::DescriptorPoolCreateInfo::default()
                            .pool_sizes(&get_pool_sizes(&self.bindings, 16))
                            .max_sets(16),
                        None,
                    )?
                };
                let mut sets = unsafe {
                    ctx.device.allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default()
                            .descriptor_pool(pool)
                            .set_layouts(&[self.handle; 16]),
                    )?
                };
                let set = sets.remove(0);
                self.pools.push(pool);
                self.sets.extend(sets);
                set
            }
        };
        Ok(set)
    }

    pub fn reclaim(&mut self, set: vk::DescriptorSet) {
        self.sets.push(set);
    }

    pub fn destroy(&mut self, ctx: &mut GpuCtx) {
        unsafe {
            ctx.device.destroy_descriptor_set_layout(self.handle, None);
            for pool in self.pools.drain(..) {
                ctx.device.destroy_descriptor_pool(pool, None);
            }
            self.sets.clear();
        }
    }
}
