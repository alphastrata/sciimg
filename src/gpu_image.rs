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

use encase::{ArrayLength, ShaderSize, ShaderType, StorageBuffer};
use glam::{Vec3, Vec3A, Vec3Swizzles, Vec4, Vec4Swizzles};
use thiserror;

pub type RGB = Vec3;
pub type RGBA = Vec4;

// #[derive(ShaderType)]
// struct MyBuffer {
//     length: ArrayLength,
//     #[size(runtime)]
//     positions: Vec<Vec4>,
// }

// #[derive(ShaderType)]
// struct MyBuffer {
//     length: ArrayLength,
//     #[size(runtime)]
//     positions: Vec<Vec3>,
// }

// A simple image raster buffer.
#[derive(ShaderType)]
pub struct GpuImage<C: ShaderSize> {
    pub width: u32,
    pub height: u32,

    // Keep in mind these must go LAST to meet WGPU's spec
    length: ArrayLength,
    /// The Vec<ImageBuffer> is inappropraite for GPUs.
    #[size(runtime)]
    pub data: Vec<C>,
}
impl<C: ShaderSize> GpuImage<C> {
    pub fn from_sciimg_image(img: &Image) -> Self {
        match img.buffers().len(){
            3 => {//RGB
                todo!()
                },
            4 => {
                //RGBA
                todo!()

            },
            _ => unreachable!("You're attempting to make a GpuImage from a sciimg::Image with insufficent channels. Only RGB and RGBA are supported"),
        }
        todo!()
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
//
//TODO:
/*
 - StorageBuffer -> wgpu::Buffer
 - Gpu::attach::<B>(&mut self, buffer: B);
*/
