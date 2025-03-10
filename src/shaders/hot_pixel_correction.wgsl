// hot_pixel_correction.wgsl
struct Params {
    width: u32,
    height: u32,
    window_size: i32,
    threshold: f32,
};

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size(16,16,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    let idx = y * i32(params.width) + x;
    
    // Skip border pixels.
    if (x < 1 || x >= i32(params.width) - 1 || y < 1 || y >= i32(params.height) - 1) {
        output[idx] = input[idx];
        return;
    }
    let half = params.window_size / 2;
    var sum: f32 = 0.0;
    var count: i32 = 0;
    // Compute mean.
    for (var j: i32 = -half; j <= half; j = j + 1) {
        for (var i: i32 = -half; i <= half; i = i + 1) {
            let nx = x + i;
            let ny = y + j;
            if (nx >= 0 && nx < i32(params.width) && ny >= 0 && ny < i32(params.height)) {
                let nidx = ny * i32(params.width) + nx;
                sum = sum + input[nidx];
                count = count + 1;
            }
        }
    }
    let mean = sum / f32(count);
    // Compute standard deviation.
    var sum_sq: f32 = 0.0;
    for (var j: i32 = -half; j <= half; j = j + 1) {
        for (var i: i32 = -half; i <= half; i = i + 1) {
            let nx = x + i;
            let ny = y + j;
            if (nx >= 0 && nx < i32(params.width) && ny >= 0 && ny < i32(params.height)) {
                let nidx = ny * i32(params.width) + nx;
                let diff = input[nidx] - mean;
                sum_sq = sum_sq + diff * diff;
            }
        }
    }
    let stddev = sqrt(sum_sq / f32(count));
    // Compute z-score.
    let z = abs(input[idx] - mean) / stddev;
    // Replace pixel if z exceeds threshold.
    output[idx] = select(input[idx], mean, z > params.threshold);
}
