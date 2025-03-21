//! The GPU version of a `sciimg::Image`
use super::image::{dimensions::ImgDimensions, Empty, GpuImage, ImageUniform};
use crate::{
    enums, image::Image, max, min, path, Dn, DnVec, Mask, MaskVec, MaskedDnVec, MinMax, VecMath,
};
use encase::{
    internal::{ReadFrom, WriteInto},
    ArrayLength, ShaderSize, ShaderType, StorageBuffer,
};
use glam::{Vec3, Vec3A, Vec3Swizzles, Vec4, Vec4Swizzles};
use wgpu::{BufferUsages, Features};

/// A `gpu` wrapper, holding all the wgpu goodies we need to get stuff done
// NOTE: You should implement things ON this.
pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,

    // work related
    pub pipeline: Option<wgpu::ComputePipeline>,
}

impl GpuContext {
    //TODO: Errors
    pub async fn new() -> Self {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("SciImg GPU Device"),
                required_features: Features::empty(),
                memory_hints: wgpu::MemoryHints::Performance,
                required_limits: wgpu::Limits::downlevel_defaults(),
                trace: wgpu::Trace::Off,
            })
            .await
            .unwrap();

        Self {
            instance,
            adapter,
            device,
            queue,
            pipeline: None,
        }
    }

    pub fn create_compute_pipeline(
        &self,
        layout: &wgpu::PipelineLayout,
        shader_module: &wgpu::ShaderModule,
        entry_point: &str,
    ) -> wgpu::ComputePipeline {
        self.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Sciimg Compute Pipeline"),
                layout: Some(layout),
                module: shader_module,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
    }
}

impl GpuContext {
    /// Host -> Device
    /// Copies a GpuImage `Into` a `wgpu::Buffer` AND writes it to GPU Storage.
    ///
    /// NOTES:
    /// * This can panic if the write to the Buffer fails.
    /// * This writes to GPU memory.
    pub fn write_img_to_device(&self, img: &GpuImage) -> (u64, wgpu::Buffer) {
        let mut input_bytes = Vec::new();
        {
            let mut sbuf = encase::StorageBuffer::new(&mut input_bytes);
            sbuf.write(img).unwrap();
        }
        let input_size = input_bytes.len() as wgpu::BufferAddress;

        let input_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GaussianBlur Input"),
            size: input_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&input_buffer, 0, &input_bytes);
        (input_size, input_buffer)
    }

    /// Host -> Device
    /// Copies `Uniforms` to the GPU.
    /// Your uniforms must derive `encase::Shadertype`
    ///
    /// NOTES:
    /// * By convention this call binds to `@group(0) @binding(0)`, if you want something other than that,
    /// you're on your own.
    /// * This can panic if the write to the Buffer fails.
    /// * This writes to GPU memory.
    pub fn write_uniforms_to_device<U>(&self, uniform_data: U) -> wgpu::Buffer
    where
        U: ShaderType + WriteInto,
    {
        let mut uniform_bytes = Vec::new();
        {
            let mut ubuf = encase::UniformBuffer::new(&mut uniform_bytes);
            ubuf.write(&uniform_data).unwrap();
        }
        let uniform_size = uniform_bytes.len() as wgpu::BufferAddress;
        let uniform_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GaussianBlur Uniform"),
            size: uniform_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&uniform_buffer, 0, &uniform_bytes);
        uniform_buffer
    }
}
