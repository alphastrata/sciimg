//! The GPU versions of Image and ImageBuffer
//!
/*

NOTE: to developers this is where most of the boilerplate can be found
https://github.com/gfx-rs/wgpu/blob/trunk/examples/features/src/storage_texture/mod.rs

*/
#![allow(unused_imports, dead_code)]
use crate::{enums, max, min, path, Dn, DnVec, Mask, MaskVec, MaskedDnVec, MinMax, VecMath};

use bytemuck::{Pod, Zeroable};

use encase::{ArrayLength, ShaderType, StorageBuffer};
use thiserror;

// A simple image raster buffer.
#[derive(ShaderType)]
pub struct GpuImageBuffer {
    pub width: u32,
    pub height: u32,

    // Keep in mind these must go LAST to meet WGPU's spec
    length: ArrayLength,
    #[size(runtime)]
    pub raw_buffer: Vec<f32>,
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
// pub struct GpuContext {
//     instance: wgpu::Instance,
//     adapter: wgpu::Adapter,
//     device: wgpu::Device,
//     queue: wgpu::Queue,
// }
//
//TODO:
/*
 - StorageBuffer -> wgpu::Buffer
 - Gpu::attach::<B>(&mut self, buffer: B);
*/
