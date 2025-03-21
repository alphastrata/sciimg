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

// const SHADER: wgpu::ShaderModuleDescriptor<'_> = include_wgsl!("../shaders/gaussian_blur.wgsl");
const SHADER: wgpu::ShaderModuleDescriptor<'_> =
    include_wgsl!("../shaders/fast_gaussian_blur.wgsl");

impl GpuContext {
    pub fn gaussian_blur(
        &self,
        img: &GpuImage,
        width: u32,
        height: u32,
        radius: u32,
        sigma: f32,
    ) -> GpuImage {
        let (output_buffer, readback_buffer, encoder) = self.inner_run_simple_gpu_job(
            img,
            width,
            height,
            GaussianBlurUniform {
                radius,
                sigma,
                width,
                height,
                pass_index: 0,
            },
            SHADER,
        );

        self.read_from_device(output_buffer, readback_buffer, encoder)
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
