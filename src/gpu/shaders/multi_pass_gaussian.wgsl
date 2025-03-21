
// First pass: horizontal blur
@compute @workgroup_size(256, 1)
fn horizontal_blur(@builtin(global_invocation_id) gid: vec3<u32>) {
  output_data.length = input_data.length;
  
  let x = i32(gid.x);
  let y = i32(gid.y);
  let w = i32(blur_params.width);
  let h = i32(blur_params.height);
  
  if (x >= w || y >= h) {
    return;
  }
  
  let rad = i32(blur_params.radius);
  let sigma2 = blur_params.sigma * blur_params.sigma;
  let two_sigma2 = 2.0 * sigma2;
  
  var sum_weights = 0.0;
  var accum = vec4<f32>(0.0);
  
  // Horizontal pass
  for (var dx = -rad; dx <= rad; dx += 1) {
    let sx = clamp(x + dx, 0, w - 1);
    let sample_idx = u32(y) * blur_params.width + u32(sx);
    
    let dist2 = f32(dx * dx);
    let weight = exp(-dist2 / two_sigma2);
    
    accum += input_data.data[sample_idx] * weight;
    sum_weights += weight;
  }
  
  output_data.data[u32(y) * blur_params.width + u32(x)] = accum / sum_weights;
}

// Second pass: vertical blur
@compute @workgroup_size(1, 256)
fn vertical_blur(@builtin(global_invocation_id) gid: vec3<u32>) {
  output_data.length = input_data.length;
  
  let x = i32(gid.x);
  let y = i32(gid.y);
  let w = i32(blur_params.width);
  let h = i32(blur_params.height);
  
  if (x >= w || y >= h) {
    return;
  }
  
  let rad = i32(blur_params.radius);
  let sigma2 = blur_params.sigma * blur_params.sigma;
  let two_sigma2 = 2.0 * sigma2;
  
  var sum_weights = 0.0;
  var accum = vec4<f32>(0.0);
  
  // Vertical pass
  for (var dy = -rad; dy <= rad; dy += 1) {
    let sy = clamp(y + dy, 0, h - 1);
    let sample_idx = u32(sy) * blur_params.width + u32(x);
    
    let dist2 = f32(dy * dy);
    let weight = exp(-dist2 / two_sigma2);
    
    accum += input_data.data[sample_idx] * weight;
    sum_weights += weight;
  }
  
  output_data.data[u32(y) * blur_params.width + u32(x)] = accum / sum_weights;
}