//! The GPU versions of Image and ImageBuffer
//!
/*

NOTE: to developers this is where most of the boilerplate can be found
https://github.com/gfx-rs/wgpu/blob/trunk/examples/features/src/storage_texture/mod.rs

*/
#![allow(unused_imports, dead_code)]
use crate::{
    enums, image::Image, max, min, path, Dn, DnVec, Mask, MaskVec, MaskedDnVec, MinMax, VecMath,
};

use bytemuck::{Pod, Zeroable};

use encase::{internal::WriteInto, ArrayLength, ShaderSize, ShaderType, StorageBuffer};
use glam::{Vec3, Vec3A, Vec3Swizzles, Vec4, Vec4Swizzles};
use thiserror;

pub type RGB = Vec3;
pub type RGBA = Vec4;

// A simple image raster buffer.
#[derive(ShaderType)]
pub struct GpuImage<C: ShaderSize + WriteInto> {
    pub width: u32,
    pub height: u32,

    /// WGPU requires this, it is the length of the below .data
    length: ArrayLength,

    /// The Vec<ImageBuffer> is inappropraite for GPUs.
    /// So we use a `Vec<Vec3>` or, a `Vec4` when there's an alpha channel.
    #[size(runtime)]
    pub data: Vec<C>,
}
impl<C: ShaderSize + WriteInto> GpuImage<C> {
    pub fn new(width: u32, height: u32, data: Vec<C>) -> Self {
        Self {
            width,
            height,
            length: ArrayLength,
            data,
        }
    }
    fn as_wgsl_bytes(&self) -> encase::internal::Result<Vec<u8>> {
        let mut buffer = encase::UniformBuffer::new(Vec::new());
        buffer.write(self)?;
        Ok(buffer.into_inner())
    }
}

impl GpuImage<Vec3> {
    pub fn from_sciimg(img: &Image) -> Self {
        let size = img.width * img.height;
        let mut data = Vec::with_capacity(size);
        for i in 0..size {
            data.push(Vec3::new(
                img.get_band(0).buffer[i],
                img.get_band(1).buffer[i],
                img.get_band(2).buffer[i],
            ));
        }

        Self {
            width: img.width as u32,
            height: img.height as u32,
            length: ArrayLength,
            data,
        }
    }
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

impl GpuImage<Vec4> {
    pub fn from_sciimg(img: &Image) -> Self {
        let size = img.width * img.height;
        let mut data = Vec::with_capacity(size);
        for i in 0..img.width * img.height {
            data.push(Vec4::new(
                img.get_band(0).buffer[i],
                img.get_band(1).buffer[i],
                img.get_band(2).buffer[i],
                if img.is_using_alpha() {
                    if img.get_alpha_at(i % img.width, i / img.width) {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    1.0
                },
            ));
        }

        Self {
            width: img.width as u32,
            height: img.height as u32,
            length: ArrayLength,
            data,
        }
    }
}

/*

On supporting enums::ImageMode
https://gpuweb.github.io/gpuweb/#texture-formats
    pub enum GpuImageMode {
        U8BIT,
        U12BIT,
        U16BIT,
    }


    There is no 12BIT support, so you can have U8 or U16

    TODO: GpuTextureFormat to match the wgpu spec our ImageBuffer into -> That.
*/

// /// A `gpu` wrapper, holding all the wgpu goodies we need to get stuff done
// // NOTE: You should implement things ON this.
pub struct GpuContext {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl GpuContext {
    pub async fn new() -> Self {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: todo!(),
                    required_limits: todo!(),
                    memory_hints: todo!(),
                },
                None,
            )
            .await
            .unwrap();

        Self {
            instance,
            adapter,
            device,
            queue,
        }
    }

    /// Load the `image` into a GPU Uniform Buffer
    /// htod (host to device)
    pub fn load_image_as_uniform<C: ShaderSize + WriteInto>(
        &self,
        image: &GpuImage<C>,
    ) -> (wgpu::Buffer, wgpu::BindGroupLayout, wgpu::BindGroup) {
        let buffer_size = (std::mem::size_of::<C>() * image.data.len()) as wgpu::BufferAddress;

        let buffer_desc = wgpu::BufferDescriptor {
            label: Some("GpuImage Uniform Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        };

        let buffer = self.device.create_buffer(&buffer_desc);

        self.queue.write_buffer(
            &buffer,
            0,
            &image
                .as_wgsl_bytes()
                .expect("Unable to write your GpuImage to GPU buffer."),
        );
        // Create bind group layout
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                    label: Some("SciImg Bind Group Layout"),
                });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
            label: Some("SciImg Bind Group"),
        });

        (buffer, bind_group_layout, bind_group)
    }
}
