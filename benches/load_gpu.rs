use criterion::{black_box, criterion_group, criterion_main, Criterion};
use sciimg::gpu_image::{GpuContext, GpuImage};

use std::time::Duration;

const INPAINT_TEST_IMAGE: &str = "tests/testdata/MSL_MAHLI_INPAINT_Sol2904_V1.png";

fn setup_benchmark_data() -> GpuImage<Vec3> {
    let img = Image::open(&String::from(INPAINT_TEST_IMAGE)).unwrap();
    GpuImage::from_sciimg(img)
}

async fn setup_gpu() -> GpuContext {
    GpuContext::new()
}

fn benchmark(c: &mut Criterion) {
    let gpu = pollster::block_on(setup_benchmark_data());
    let img = setup_benchmark_data();

    gpu.load_image_as_uniform(&img);

    //TODO:
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
