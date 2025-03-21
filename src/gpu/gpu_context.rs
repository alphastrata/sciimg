//! The GPU version of a `sciimg::Image`
use super::image::{Empty, GpuImage};
use encase::{internal::WriteInto, ShaderType};
use wgpu::Features;

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

    fn create_compute_pipeline(
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

    /// Host -> Device
    /// Copies a GpuImage `Into` a `wgpu::Buffer` AND writes it to GPU Storage.
    ///
    /// NOTES:
    /// * This can panic if the write to the Buffer fails.
    /// * This writes to GPU memory.
    fn write_img_to_device(&self, img: &GpuImage) -> (u64, wgpu::Buffer) {
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
    fn write_uniforms_to_device<U>(&self, uniform_data: U) -> wgpu::Buffer
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

    /// In Sciimg out GPU accelerated image ops are built around a very simple flow
    /// TODO: ascii art of how we have one input image, one output image, and one set of uniforms that we leave
    /// for developers to have freedom over.
    /// We do this to guarantee (for beginners etc) that shader bindings if they follow existing code:
    /// ```rust,ignore
    ///    @group(0) @binding(0) var<uniform> blur_params: GaussianBlurUniform;
    ///    @group(0) @binding(1) var<storage, read> input_data: GpuImg;
    ///    @group(1) @binding(0) var<storage, read_write> output_data: GpuImg;
    /// ```
    /// Will work for them.
    fn setup_bindgroups_and_layouts(
        &self,
        input_buffer: wgpu::Buffer,
        output_buffer: &wgpu::Buffer,
        uniform_buffer: wgpu::Buffer,
    ) -> (
        wgpu::BindGroupLayout,
        wgpu::BindGroupLayout,
        wgpu::BindGroup,
        wgpu::BindGroup,
    ) {
        // @group(0)
        let group0 = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("GaussianBlur Inputs & Uniforms"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // @group(1)
        let group1 = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Sciimg ReadBack"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        // 0
        let group0_binds = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sciimg bind_group0"),
            layout: &group0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_buffer.as_entire_binding(),
                },
            ],
        });

        // 1
        let group1_binds = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sciimg bind_group1"),
            layout: &group1,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: output_buffer.as_entire_binding(),
            }],
        });
        (group0, group1, group0_binds, group1_binds)
    }

    fn create_encoder(&self) -> wgpu::CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Sciimg Compute Encoder"),
            })
    }
    /// Sets up the bindings the input `compute_pass` will use, i.e, creates these:
    /// ```rust,ignore
    ///     @group(0) @binding(x) ...
    ///     @group(1) @binding(y) ...
    ///```
    fn set_binds<'buf, I>(&self, compute_pass: &mut wgpu::ComputePass<'_>, bind_groups: I)
    where
        I: IntoIterator<Item = &'buf wgpu::BindGroup>,
    {
        let mut idx: u32 = 0;
        for bind_group in bind_groups.into_iter() {
            compute_pass.set_bind_group(idx, bind_group, &[]);
            idx += 1;
        }
    }

    /// Creates a pipeline layout suitable for 2, and ONLY 2 Layouts.
    fn create_pipeline_layout(
        &self,
        bind_group_layouts: &[&wgpu::BindGroupLayout; 2],
        shader: wgpu::ShaderModuleDescriptor,
    ) -> (wgpu::PipelineLayout, wgpu::ShaderModule) {
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("GaussianBlur pipeline layout"),
                bind_group_layouts,
                push_constant_ranges: &[],
            });
        let cs_module = self.device.create_shader_module(shader);

        (pipeline_layout, cs_module)
    }
    fn create_compute_pass<'e>(
        &self,
        pipeline: wgpu::ComputePipeline,
        encoder: &'e mut wgpu::CommandEncoder,
    ) -> wgpu::ComputePass<'e> {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Sciimg Compute Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);

        compute_pass
    }

    /// Dispatches a GPU compute job.
    ///
    /// `width` and `height` define the total work area. `x_div_ceil` and `y_div_ceil`
    /// control workgroup size (defaulting to 16x16). `div_ceil` ensures all data is
    /// processed, rounding up the number of workgroups if `width` or `height` are not
    /// perfectly divisible by the workgroup size. The shader must handle out-of-bounds
    /// access.
    fn run_compute_job(
        &self,
        width: u32,
        height: u32,
        x_div_ceil: Option<u32>,
        y_div_ceil: Option<u32>,
        mut compute_pass: wgpu::ComputePass<'_>,
    ) {
        let gx = width.div_ceil(x_div_ceil.unwrap_or(16));
        let gy = height.div_ceil(y_div_ceil.unwrap_or(16));
        compute_pass.dispatch_workgroups(gx, gy, 1);
        drop(compute_pass); //TODO: drop ain't magic as the docs say do we need this if we've wrapped it up in a func?
    }

    /// Device -> Host
    ///
    /// This copies BACK the GpuImage you sent over, i.e you've called `GpuContext::run_compute_job`
    /// Call this AFTER your processing is done (although note that it blocks).
    ///
    /// NOTES:
    /// - Panics if the output is empty.
    pub(crate) fn read_from_device(
        &self,
        output_buffer: wgpu::Buffer,
        readback_buffer: wgpu::Buffer,
        mut encoder: wgpu::CommandEncoder,
    ) -> GpuImage {
        // 10) enque a copy Device -> Host & submit it.
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, output_buffer.size());
        let submit = encoder.finish();
        self.queue.submit([submit]);

        // 11) Read stuff back n wait...
        let buffer_slice = readback_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| ());
        self.device.poll(wgpu::PollType::Wait).unwrap();

        // 12) Map (DtoH) results
        let mapped_range = buffer_slice.get_mapped_range();
        let mut final_bytes = mapped_range.to_vec();
        drop(mapped_range);
        readback_buffer.unmap();

        assert!(
            !final_bytes.is_empty(),
            "Failed to copy any data back from the Shader's output..."
        );

        // 13) Decode
        let sbuf = encase::StorageBuffer::new(&mut final_bytes);
        let mut new_image: GpuImage = GpuImage::empty();
        match sbuf.read(&mut new_image) {
            Ok(_) => {
                assert!(!new_image.data.is_empty());
                log::trace!("Successfully read data: {} elements", new_image.data.len())
            }
            Err(e) => panic!("Failed to deserialize buffer: {:?}", e),
        };

        new_image
    }
}

impl GpuContext {
    fn default_io_buffers(
        &self,
        input_size: u64,
        input_buffer: &wgpu::Buffer,
    ) -> (wgpu::Buffer, wgpu::Buffer) {
        // NOTE: Don't abstract these -- it's not worth it to have a 'helper'
        // 2a) Output buffer
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GaussianBlur Output"),
            size: input_buffer.size(), // Must be the same
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // 2b) we make a final buffer that can be read FROM the cpu
        let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GaussianBlur Staging"),
            size: input_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        (output_buffer, readback_buffer)
    }

    /// If you're happy to run with the defaults Sciimg has decided, then this is probably the way
    /// to go for your first few image processing shaders.
    ///
    /// It should be a safe starting point to wet your feet with image processing on GPUs.
    ///
    /// This call makes many many assumptions, basically in an effort to make a simpler UI we're removing all the
    /// pipeline control.
    pub(crate) fn inner_run_simple_gpu_job<U: ShaderType + WriteInto>(
        &self,
        img: &GpuImage,
        width: u32,
        height: u32,
        uniform_data: U,
        shader: wgpu::ShaderModuleDescriptor,
    ) -> (wgpu::Buffer, wgpu::Buffer, wgpu::CommandEncoder) {
        let (input_size, input_buffer) = self.write_img_to_device(img);
        let (output_buffer, readback_buffer) = self.default_io_buffers(input_size, &input_buffer);
        let uniform_buffer = self.write_uniforms_to_device(uniform_data);

        let (group0_layout, group1_layout, group0_binds, group1_binds) =
            self.setup_bindgroups_and_layouts(input_buffer, &output_buffer, uniform_buffer);

        let (pipeline_layout, cs_module) =
            self.create_pipeline_layout(&[&group0_layout, &group1_layout], shader);
        let pipeline = self.create_compute_pipeline(&pipeline_layout, &cs_module, "main");

        let mut encoder = self.create_encoder();
        let mut compute_pass = self.create_compute_pass(pipeline, &mut encoder);
        self.set_binds(&mut compute_pass, [&group0_binds, &group1_binds]);

        self.run_compute_job(width, height, None, None, compute_pass);

        (output_buffer, readback_buffer, encoder)
    }
}
