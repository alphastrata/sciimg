// Horizontal pass
struct GaussianBlurParams {
  radius: u32,
  sigma: f32,
  width: u32,
  height: u32,
  isHorizontal: u32, 
};

struct GpuImg {
  length: u32,
  data: array<vec4<f32>>,
};

@group(0) @binding(0) var<uniform> blur_params: GaussianBlurUniform;
@group(0) @binding(1) var<storage, read> input_data: GpuImg;
@group(1) @binding(0) var<storage, read_write>  output_data: GpuImg;

// Workgroup size optimized for better occupancy
@compute @workgroup_size(256, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  output_data.length = input_data.length;
  
  let x = i32(gid.x);
  let y = i32(gid.y);
  let w = i32(blur_params.width);
  let h = i32(blur_params.height);
  
  // Early exit if out of bounds
  if (x >= w || y >= h) {
    return;
  }
  
  let rad = i32(blur_params.radius);
  let sigma2 = blur_params.sigma * blur_params.sigma;
  let two_sigma2 = 2.0 * sigma2;
  
  var sum_weights = 0.0;
  var accum = vec4<f32>(0.0);
  
  // Apply either horizontal or vertical pass based on parameter
  if (blur_params.isHorizontal == 1u) {
    // Horizontal pass
    for (var dx = -rad; dx <= rad; dx += 1) {
      let sx = clamp(x + dx, 0, w - 1);
      let sample_idx = u32(y) * blur_params.width + u32(sx);
      
      let dist2 = f32(dx * dx);
      let weight = exp(-dist2 / two_sigma2);
      
      accum += input_data.data[sample_idx] * weight;
      sum_weights += weight;
    }
  } else {
    // Vertical pass
    for (var dy = -rad; dy <= rad; dy += 1) {
      let sy = clamp(y + dy, 0, h - 1);
      let sample_idx = u32(sy) * blur_params.width + u32(x);
      
      let dist2 = f32(dy * dy);
      let weight = exp(-dist2 / two_sigma2);
      
      accum += input_data.data[sample_idx] * weight;
      sum_weights += weight;
    }
  }
  
  // Output normalized result
  output_data.data[u32(y) * blur_params.width + u32(x)] = accum / sum_weights;
}