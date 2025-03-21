//! image processing
use crate::gpu::{gpu_context::GpuContext, image::GpuImage};
use encase::ShaderType;
use wgpu::include_wgsl;

#[derive(ShaderType)]
struct GaussianBlurUniform {
    pub radius: u32,
    pub sigma: f32,
    pub width: u32,
    pub height: u32,
    pub pass_index: u32,
}
impl GpuContext {
    const SHADER: wgpu::ShaderModuleDescriptor<'_> = include_wgsl!("../shaders/gaussian_blur.wgsl");

    pub fn gaussian_blur(
        &self,
        img: &GpuImage,
        width: u32,
        height: u32,
        radius: u32,
        sigma: f32,
    ) -> GpuImage {
        let uniform_data = GaussianBlurUniform {
            radius,
            sigma,
            width,
            height,
            pass_index: 0,
        };

        // 1) Create & fill input buffer
        let (input_size, input_buffer) = self.write_img_to_device(img);

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

        // Uniforms are special:
        // 3) Uniform buffer
        let uniform_buffer = self.write_uniforms_to_device(uniform_data);

        // 4) Bind groups & Layouts for them.
        let (group0_layout, group1_layout, group0_binds, group1_binds) =
            self.setup_bindgroups_and_layouts(input_buffer, &output_buffer, uniform_buffer);

        // 6) Pipeline
        let (pipeline_layout, cs_module) =
            self.create_pipeline_layout(&[&group0_layout, &group1_layout], Self::SHADER);
        // .create_shader_module(wgpu::include_wgsl!("../shaders/fast_gaussian_blur.wgsl"));

        let pipeline = self.create_compute_pipeline(&pipeline_layout, &cs_module, "main");

        // 7) Bind things TO that Pipeline
        let mut encoder = self.create_encoder();

        // 8) Build a compute pass for the Pipeline, bind the binds to it.
        let mut compute_pass = self.create_compute_pass(pipeline, &mut encoder);

        let bind_groups = [&group0_binds, &group1_binds];
        self.set_binds(&mut compute_pass, bind_groups);

        // 9) Dispatch the work! ** Actually run shit on the GPU **
        self.run_compute_job(width, height, None, None, compute_pass);

        self.readback_gpu(output_buffer, readback_buffer, encoder)
    }
}

#[cfg(test)]
mod test {
    use crate::image::Image;

    use super::*;
    const INPAINT_TEST_IMAGE: &str =
        "tests/testdata/ZL0_0038_0670307360_057ECM_N0031392ZCAM08007_1100LUJ.png";

    #[test]
    fn gpu_gaussian_blur() {
        // _ = pretty_env_logger::init();
        let gpu = pollster::block_on(GpuContext::new());
        let start_img = Image::open(&String::from(INPAINT_TEST_IMAGE)).unwrap();
        let gpu_img = GpuImage::from_sciimg_rgb(&start_img);

        let radius = 2;
        let sigma = 2.8;
        let (width, height) = (start_img.width, start_img.height);

        let res = gpu.gaussian_blur(&gpu_img, width as u32, height as u32, radius, sigma);
        assert_eq!(start_img.get_band(0).buffer.len(), res.data.len(),);
        let res_as_sciimg = res.to_sciimg(width, height).unwrap();

        assert_eq!(
            start_img.get_band(0).buffer.len(),
            res_as_sciimg.get_band(0).buffer.len(),
        );
        let zeroes = glam::Vec4::ZERO.to_array();
        assert!(res.data.iter().all(|v| v.to_array() != zeroes));
    }
}
