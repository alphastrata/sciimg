use crate::{image::Image, prelude::ImageBuffer, Dn};
use flume;
use std::default;
use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: u32 = 3;

pub struct SciImgGpuWrapper {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct Params {
    width: usize,
    height: usize,
    window_size: i32, // fixed to 3
    threshold: f32,
}
unsafe impl bytemuck::Pod for Params {}
unsafe impl bytemuck::Zeroable for Params {}

impl SciImgGpuWrapper {
    pub async fn new() -> Result<Self, anyhow::Error> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("SciImg GPU device"),
                    memory_hints: wgpu::MemoryHints::Performance,
                    required_limits: wgpu::Limits::default(),
                    ..Default::default()
                },
                None,
            )
            .await?;
        Ok(Self { device, queue })
    }

    pub async fn hot_pixel_correction(
        &self,
        img: &mut Image,
        threshold: f32,
    ) -> Result<(), anyhow::Error> {
        // Convert image to f32 vec.
        let numbers = img.to_f32_vec();
        let buffer_size = (numbers.len() * std::mem::size_of::<f32>()) as wgpu::BufferAddress;

        // Create input & output buffers.
        let input_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Input Buffer"),
                contents: bytemuck::cast_slice(&numbers),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Load compute shader.
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Hot Pixel Correction Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("shaders/hot_pixel_correction.wgsl").into(),
                ),
            });

        let image_width = img.width;
        let image_height = img.height;

        let params_data = Params {
            width: image_width,
            height: image_height,
            window_size: 3,
            threshold,
        };
        let uniform_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Uniform Buffer"),
                contents: bytemuck::bytes_of(&params_data),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create bind group layout.
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("HPC Bind Group Layout"),
                    entries: &[
                        // Input (read-only)
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Output (read-write)
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Uniform
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        // Create pipeline layout & compute pipeline.
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
        let compute_pipeline =
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Compute Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        // Create bind group.
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("HPC Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("HPC Encoder"),
            });

        // Dispatch workgroups (ceil division for each dimension).
        let workgroups_x = ((image_width as u32) + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        let workgroups_y = ((image_height as u32) + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("HPC Compute Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, buffer_size);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let data = {
            let mapped = buffer_slice.get_mapped_range();
            bytemuck::cast_slice(&mapped).to_vec()
        };
        staging_buffer.unmap();
        img.from_f32_vec(&data);
        Ok(())
    }
}

impl crate::image::Image {
    pub fn iter_inner_buffer_dn(&self) -> std::slice::Iter<'_, ImageBuffer> {
        self.bands.iter()
    }
    pub fn to_f32_vec(&self) -> Vec<Dn> {
        // Use cloned() to convert &f32 to f32.
        self.bands.iter().flat_map(|b| b.buffer.iter()).collect()
    }
    pub fn from_f32_vec(&mut self, data: &[Dn]) {
        let buf = ImageBuffer::from_vec(data, self.width, self.height)
            .expect("failed to create an ImageBuffer from data.");

        std::mem::swap(&mut self.bands, &mut vec![buf]);
    }
}

#[cfg(test)]
mod tests {
    use super::SciImgGpuWrapper;
    use crate::image::Image;

    #[test]
    fn hot_pixel_correction() {
        let test_image_path = "tests/testdata/MSL_MAHLI_INPAINT_Sol2904_V1.png";
        let mut img = Image::open(test_image_path).unwrap();
        dbg!(img.width, img.height);

        let non_zero_before = img
            .bands
            .iter()
            .flat_map(|b| b.buffer.iter())
            .filter(|&x| x != 0.0)
            .count();

        dbg!(&non_zero_before);
        // img.hot_pixel_correction(3, 1.0); // Adjust params if needed

        // let non_zero_after = img
        //     .bands
        //     .iter()
        //     .flat_map(|b| b.buffer.iter())
        //     .filter(|&x| x != 0.0)
        //     .count();
        // dbg!(&non_zero_after);

        // assert_ne!(non_zero_before, non_zero_after);
    }

    #[test]
    fn gpu_hot_px_correction() {
        let test_image_path = "tests/testdata/MSL_MAHLI_INPAINT_Sol2904_V1.png";
        let mut img = Image::open(test_image_path).unwrap();
        pollster::block_on(async {
            let gpu = SciImgGpuWrapper::new().await.unwrap();
            let threshold = 1.0;
            _ = gpu.hot_pixel_correction(&mut img, threshold).await.unwrap();
        });
    }
}
