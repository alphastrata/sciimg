struct GaussianBlurUniform {
    radius : u32,
    sigma  : f32,
    width  : u32,
    height : u32,
    pass_index: u32,
};

struct GpuImg {
    length: u32,
    data: array<vec4<f32>>,
};

@group(0) @binding(0) var<uniform> blur_params: GaussianBlurUniform;
@group(0) @binding(1) var<storage, read> input_data: GpuImg;
@group(1) @binding(0) var<storage, read_write> output_data: GpuImg;

// Helper function to calculate box sizes for Gaussian approximation
fn boxes_for_gauss(sigma: f32, n: u32) -> array<i32, 3> {
    // Ideal box width for given sigma and number of boxes
    // FIXME: why 12?
    let ideal_width = sqrt(12.0 * sigma * sigma / f32(n) + 1.0);
    var width_up = i32(floor(ideal_width));
    if (width_up % 2 == 0) {
        width_up = width_up - 1;
    }
    
    let width_down = width_up - 2;
    
     // FIXME: why 12?
    let m_ideal = (12.0 * sigma * sigma - f32(n) * f32(width_down * width_down) - 4.0 * f32(n) * f32(width_down) - 3.0 * f32(n)) / (-4.0 * f32(width_down) - 4.0);
    let m = i32(round(m_ideal));
    
    var sizes: array<i32, 3>;
    for (var i = 0u; i < n; i = i + 1u) {
        if (i < u32(m)) {
            sizes[i] = width_up;
        } else {
            sizes[i] = width_down;
        }
    }
    
    return sizes;
}

// Main compute shader for fast Gaussian blur
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    output_data.length = input_data.length;
    
    let x = i32(gid.x);
    let y = i32(gid.y);
    let w = i32(blur_params.width);
    let h = i32(blur_params.height);
    
    if (x >= w || y >= h) { return; }
    
    // Calculate box radii for 3 passes (standard Gaussian approximation)
    let sigma = blur_params.sigma;
    //TODO: const
    let boxes = boxes_for_gauss(sigma, 3u);
    
    // Determine which box radius to use based on the current pass
    let r = (boxes[blur_params.pass_index % 3u] - 1) / 2;
    
    // Determine if we're doing horizontal or vertical pass
    let is_horizontal = (blur_params.pass_index / 3u) % 2u == 0u;
    
    if (is_horizontal) {
        // Horizontal pass
        horizontal_blur(gid, i32(r));
    } else {
        // Vertical pass
        vertical_blur(gid, i32(r));
    }
}

// Fast horizontal box blur using accumulator method
fn horizontal_blur(gid: vec3<u32>, r: i32) {
    let y = i32(gid.y);
    let w = i32(blur_params.width);
    let h = i32(blur_params.height);
    
    if (y >= h) { return; }
    
    // Inverse of diameter (for normalization)
    let iarr = 1.0 / f32(r + r + 1);
    
    // Row start index
    let row_start = u32(y) * blur_params.width;
    
    // Initialize positions
    var ti = row_start; // Target index
    var li = ti;        // Left index
    var ri = min(ti + u32(r), row_start + blur_params.width - 1u); // Right index (bounded)
    
    // First value in row
    let fv = input_data.data[row_start];
    // Last value in row
    let lv = input_data.data[row_start + blur_params.width - 1u];
    
    // Initialize accumulator with first value multiplied by (r+1)
    var val = fv * f32(r + 1);
    
    // Add first r elements to accumulator (or as many as available)
    for (var j = 0; j < r && j < w; j = j + 1) {
        val = val + input_data.data[row_start + u32(j)];
    }
    
    // First part: add right element, subtract nothing
    for (var j = 0; j <= r && j < w; j = j + 1) {
        val = val + input_data.data[ri] - fv;
        output_data.data[ti] = val * iarr;
        ri = min(ri + 1u, row_start + blur_params.width - 1u);
        ti = ti + 1u;
    }
    
    // Middle part: add right, subtract left
    for (var j = r + 1; j < w - r; j = j + 1) {
        val = val + input_data.data[ri] - input_data.data[li];
        output_data.data[ti] = val * iarr;
        ri = ri + 1u;
        li = li + 1u;
        ti = ti + 1u;
    }
    
    // Last part: add nothing, subtract left
    for (var j = max(r + 1, w - r); j < w; j = j + 1) {
        val = val + lv - input_data.data[li];
        output_data.data[ti] = val * iarr;
        li = li + 1u;
        ti = ti + 1u;
    }
}

// Fast vertical box blur using accumulator method
fn vertical_blur(gid: vec3<u32>, r: i32) {
    let x = i32(gid.x);
    let w = i32(blur_params.width);
    let h = i32(blur_params.height);
    
    if (x >= w) { return; }
    
    // Inverse of diameter (for normalization)
    let iarr = 1.0 / f32(r + r + 1);
    
    // Starting column position
    var ti = u32(x);     // Target index
    var li = ti;         // Top index
    var ri = min(ti + u32(r * w), u32(x) + blur_params.width * (blur_params.height - 1u)); // Bottom index (bounded)
    
    // First value in column
    let fv = input_data.data[ti];
    // Last value in column
    let lv = input_data.data[u32(x) + blur_params.width * (blur_params.height - 1u)];
    
    // Initialize accumulator with first value multiplied by (r+1)
    var val = fv * f32(r + 1);
    
    // Add first r elements to accumulator (or as many as available)
    for (var j = 0; j < r && j < h; j = j + 1) {
        val = val + input_data.data[ti + u32(j * w)];
    }
    
    // First part: add bottom element, subtract nothing
    for (var j = 0; j <= r && j < h; j = j + 1) {
        val = val + input_data.data[ri] - fv;
        output_data.data[ti] = val * iarr;
        ri = min(ri + blur_params.width, u32(x) + blur_params.width * (blur_params.height - 1u));
        ti = ti + blur_params.width;
    }
    
    // Middle part: add bottom, subtract top
    for (var j = r + 1; j < h - r; j = j + 1) {
        val = val + input_data.data[ri] - input_data.data[li];
        output_data.data[ti] = val * iarr;
        ri = ri + blur_params.width;
        li = li + blur_params.width;
        ti = ti + blur_params.width;
    }
    
    // Last part: add nothing, subtract top
    for (var j = max(r + 1, h - r); j < h; j = j + 1) {
        val = val + lv - input_data.data[li];
        output_data.data[ti] = val * iarr;
        li = li + blur_params.width;
        ti = ti + blur_params.width;
    }
}