use crate::{image::Image, prelude::ImageBuffer, Dn};
use std::collections::HashMap;
use wgpu::util::DeviceExt;

// Increased for better GPU occupancy
const WORKGROUP_SIZE: u32 = 16;

/// 3*3 grid.
const HOT_PX_CORRECTION_TILE_DIM: i32 = 3;

/// A set of reusable buffers for a specific size
struct BufferSet {
    input_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    size: usize,
}

// struct WgpuContext {
//     device: wgpu::Device,
//     queue: wgpu::Queue,
//     pipeline: wgpu::ComputePipeline,
//     bind_group: wgpu::BindGroup,
//     storage_buffer: wgpu::Buffer,
//     output_staging_buffer: wgpu::Buffer,
// }

/// Wrapping a `[wgpu::Device]`, which is effectively a `GPU` handle.
pub struct SciImgGpuWrapper {
    device: wgpu::Device,
    queue: wgpu::Queue,
    // Buffer cache keyed by size for reuse
    buffer_pool: HashMap<usize, BufferSet>,
}

/// This is an exact copy of the struct we read in the hot_pixel_correction.wgsl shader.
/// Keep it in sync!
#[repr(C)]
#[derive(Copy, Clone)]
struct HotPxCorrectionParams {
    /// Image Width
    width: usize,
    /// Image Height
    height: usize,
    /// This is ALWAYS square (for our purposes)
    tile_size: i32,
    /// Values under this will NOT be corrected.
    ///
    /// Recommended values are typically between 0.5 and 3.0 depending on the image noise level.
    /// Lower values will correct more pixels but may affect valid bright spots.
    threshold: f32,
}
unsafe impl bytemuck::Pod for HotPxCorrectionParams {}
unsafe impl bytemuck::Zeroable for HotPxCorrectionParams {}

impl SciImgGpuWrapper {
    pub async fn new() -> Result<Self, anyhow::Error> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .ok_or_else(|| anyhow::anyhow!("Failed to find an appropriate GPU adapter"))?;

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

        Ok(Self {
            device,
            queue,
            buffer_pool: HashMap::new(),
        })
    }

    /// Get or create a buffer set for the specified size
    /// Returns a reference to the created or existing BufferSet
    fn ensure_buffer_set(&mut self, size: usize) -> &BufferSet {
        if !self.buffer_pool.contains_key(&size) {
            let buffer_address = size as wgpu::BufferAddress;

            let input_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Reusable Input Buffer"),
                size: buffer_address,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Reusable Output Buffer"),
                size: buffer_address,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Reusable Staging Buffer"),
                size: buffer_address,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            self.buffer_pool.insert(
                size,
                BufferSet {
                    input_buffer,
                    output_buffer,
                    staging_buffer,
                    size,
                },
            );
        }

        self.buffer_pool.get(&size).unwrap()
    }

    /// Clean up the buffer pool, releasing GPU resources
    pub fn cleanup_buffers(&mut self) {
        self.buffer_pool.clear();
    }

    /// Apply hot pixel correction to an image
    ///
    /// # Parameters
    /// * `img` - The image to process, note this call mutates the inner data.
    /// * `threshold` - Sensitivity threshold for hot pixel detection.
    pub async fn hot_pixel_correction(
        &mut self,
        img: &mut Image,
        threshold: f32,
    ) -> Result<(), anyhow::Error> {
        let raw_px_data = img.to_f32_vec();
        #[cfg(debug_assertions)]
        log::debug!("px count = {}", raw_px_data.len());

        let buffer_size = raw_px_data.len() * std::mem::size_of::<f32>();

        self.ensure_buffer_set(buffer_size);
        let buffer_set = self.buffer_pool.get(&buffer_size).unwrap();

        self.queue.write_buffer(
            &buffer_set.input_buffer,
            0,
            bytemuck::cast_slice(&raw_px_data),
        );

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

        let params_data = HotPxCorrectionParams {
            width: image_width,
            height: image_height,
            tile_size: HOT_PX_CORRECTION_TILE_DIM,
            threshold,
        };

        let uniform_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Uniform Buffer"),
                contents: bytemuck::bytes_of(&params_data),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // NOTE: bindings MUST correspond to the values as declared in the shader(s) being used.
        // take note of the `binding: 1` etc values and note that they correspond to structs shaderside.
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

        // Create pipeline layout
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        // Create compute pipeline
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

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("HPC Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_set.input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_set.output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        // Create command encoder
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("HPC Encoder"),
            });

        // Dispatch workgroups (ceil division for each dimension)
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

        // Copy from the output buffer to the staging buffer
        encoder.copy_buffer_to_buffer(
            &buffer_set.output_buffer,
            0,
            &buffer_set.staging_buffer,
            0,
            buffer_size as wgpu::BufferAddress,
        );

        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = buffer_set.staging_buffer.slice(..);
        let (sender, receiver) = flume::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        self.device.poll(wgpu::Maintain::Wait);

        receiver.recv_async().await.unwrap()?;
        drop(receiver);

        let data = {
            let mapped = buffer_slice.get_mapped_range();
            bytemuck::cast_slice(&mapped).to_vec()
        };

        assert_ne!(data, raw_px_data);

        buffer_set.staging_buffer.unmap();

        img.set_bands_from_raw(data);

        Ok(())
    }
}

impl Drop for SciImgGpuWrapper {
    fn drop(&mut self) {
        // Ensure all GPU resources are properly released
        self.cleanup_buffers();
        // Let the device finalize any pending work
        self.device.poll(wgpu::Maintain::Wait);
    }
}

impl crate::image::Image {
    pub fn to_f32_vec(&self) -> Vec<Dn> {
        self.bands.iter().flat_map(|b| b.buffer.iter()).collect()
    }

    /// Swaps over the values in self.bands with new ones from the input `data`
    /// NOTE: this panics if the data is not in the right shape etc.
    pub fn set_bands_from_raw(&mut self, data: Vec<Dn>) {
        //Calculate pixels per channel
        //NOTE: self.bands()'s ImageBuffer is actually the complete image, just for ONE of its `n` channels.
        let channel_size = self.width * self.height;
        assert_eq!(
            data.len() % channel_size,
            0,
            "Data length must be a multiple of width*height"
        );
        let channel_count = data.len() / channel_size;

        self.bands = (0..channel_count)
            .map(|i| {
                let start = i * channel_size;
                let end = start + channel_size;

                //NOTE: intentional panic
                ImageBuffer::from_vec(&data[start..end], self.width, self.height)
                    .expect("failed to create an ImageBuffer from data")
            })
            .collect();
    }
}

#[cfg(test)]
mod tests {
    use super::SciImgGpuWrapper;
    use crate::{image::Image, Dn};

    #[test]
    fn gpu_hot_px_correction() {
        _ = pretty_env_logger::init();
        let test_image_path = "tests/testdata/MSL_MAHLI_INPAINT_Sol2904_V1.png";
        // Load the original image from disk as the baseline.
        let baseline = Image::open(test_image_path).unwrap();
        // Clone baseline for CPU and GPU processing.
        let mut cpu_img = baseline.clone();
        let mut gpu_img = baseline.clone();

        // Process CPU image.
        cpu_img.hot_pixel_correction(3, 1.0);
        // Process GPU image.
        pollster::block_on(async {
            let mut gpu = SciImgGpuWrapper::new().await.unwrap();
            gpu.hot_pixel_correction(&mut gpu_img, 1.0).await.unwrap();
        });

        // Compute per-pixel differences (e.g. count pixels that changed significantly).
        let diff_baseline_cpu = pixel_diff_count(&baseline, &cpu_img);
        let diff_baseline_gpu = pixel_diff_count(&baseline, &gpu_img);
        let diff_cpu_gpu = pixel_diff_count(&cpu_img, &gpu_img);

        log::debug!("Baseline vs CPU: {} changed pixels", diff_baseline_cpu);
        log::debug!("Baseline vs GPU: {} changed pixels", diff_baseline_gpu);
        log::debug!("CPU vs GPU difference: {} pixels", diff_cpu_gpu);

        // For example, assert that CPU and GPU differences are within 20% of the baseline change count.
        assert!(
            (diff_cpu_gpu as f32) / ((diff_baseline_cpu as f32) + 1.0) < 0.2,
            "CPU and GPU implementations differ by more than 20% relative to baseline corrections"
        );
    }

    // New pixel_diff_count: count a pixel (x,y) as changed if any channel differs beyond epsilon.
    fn pixel_diff_count(img_a: &Image, img_b: &Image) -> usize {
        let width = img_a.width;
        let height = img_a.height;
        let num_pixels = width * height;
        let num_channels = img_a.bands.len();
        let data_a = img_a.to_f32_vec();
        let data_b = img_b.to_f32_vec();
        let channel_size = num_pixels; // Each band is one channel.
        let epsilon = 0.001;
        let mut count = 0;
        for pixel in 0..num_pixels {
            let mut changed = false;
            for channel in 0..num_channels {
                let idx = pixel + channel * channel_size;
                if (data_a[idx] - data_b[idx]).abs() > epsilon {
                    changed = true;
                    break;
                }
            }
            if changed {
                count += 1;
            }
        }
        count
    }

    #[test]
    fn hot_pixel_correction() {
        let test_image_path = "tests/testdata/MSL_MAHLI_INPAINT_Sol2904_V1.png";
        let original_img = Image::open(test_image_path).unwrap();
        let mut img = original_img.clone();

        // Save original pixel values for quality validation
        let original_values: Vec<Dn> = original_img.to_f32_vec();

        img.hot_pixel_correction(3, 1.0);

        // Get processed values
        let corrected_values = img.to_f32_vec();

        // Count pixels that changed
        let changed_pixels = original_values
            .iter()
            .zip(corrected_values.iter())
            .filter(|(&orig, &new)| (orig - new).abs() > 0.001)
            .count();

        // Find maximum change in any pixel
        let max_change = original_values
            .iter()
            .zip(corrected_values.iter())
            .map(|(orig, new)| (orig - new).abs())
            .fold(0.0, |max, change| if change > max { change } else { max });

        // There should be some changes and they should be significant
        assert!(
            changed_pixels > 0,
            "No pixels were changed by hot pixel correction"
        );
        assert!(
            max_change > 0.1,
            "No significant pixel corrections detected"
        );
    }
}
