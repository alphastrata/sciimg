// hot_pixel_correction.wgsl
//
// This version does per-channel hot pixel detection & correction in the same way as
// the CPU code: it computes the mean/stddev on a single channel's window and, if the
// pixel is above threshold, replaces that channel's pixel with the channel's mean.
//
struct Params {
    width: u32,
    height: u32,
    window_size: i32,
    threshold: f32,
};

@group(0) @binding(0)
var<storage, read> input: array<f32>;      // Flattened as [R[...], G[...], B[...]]
@group(0) @binding(1)
var<storage, read_write> output: array<f32>;
@group(0) @binding(2)
var<uniform> params: Params;

fn load_pixel(idx: i32, channelSize: i32) -> vec3<f32> {
    return vec3<f32>(
        input[idx],
        input[idx + channelSize],
        input[idx + 2 * channelSize]
    );
}

fn store_pixel(idx: i32, channelSize: i32, col: vec3<f32>) {
    output[idx] = col.r;
    output[idx + channelSize] = col.g;
    output[idx + 2 * channelSize] = col.b;
}

// Helper: read one channel's value at global pixel idx.
fn load_channel(idx: i32, channelSize: i32, channel: i32) -> f32 {
    return input[idx + channel * channelSize];
}

// Helper: write one channel's corrected value to global pixel idx.
fn store_channel(idx: i32, channelSize: i32, channel: i32, val: f32) {
    output[idx + channel * channelSize] = val;
}

// Accumulate mean/stddev for a single channel in a window around (x,y).
fn process_channel(idx: i32, channelSize: i32, x: i32, y: i32, channel: i32) -> f32 {
    let w = i32(params.width);
    let h = i32(params.height);
    let half = params.window_size / 2;

    var sum: f32 = 0.0;
    var sumSq: f32 = 0.0;
    var count: i32 = 0;

    for (var dy = -half; dy <= half; dy = dy + 1) {
        for (var dx = -half; dx <= half; dx = dx + 1) {
            let nx = x + dx;
            let ny = y + dy;
            if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                let nidx = ny * w + nx;
                let val = load_channel(nidx, channelSize, channel);
                sum = sum + val;
                sumSq = sumSq + (val * val);
                count = count + 1;
            }
        }
    }

    let mean = sum / f32(count);
    let variance = (sumSq / f32(count)) - (mean * mean);
    let stddev = sqrt(max(variance, 0.0));
    let currVal = load_channel(idx, channelSize, channel);

    // z-score
    let z = select(0.0, abs(currVal - mean) / stddev, stddev > 0.0);

    // If the pixel is above threshold, replace with mean
     if (z > params.threshold) { 
        return mean; 
    } else { 
        return currVal; 
    }
}

@compute @workgroup_size(16,16,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = i32(params.width);
    let height = i32(params.height);
    let x = i32(gid.x);
    let y = i32(gid.y);

    // Guard if out of bounds
    if (x >= width || y >= height) {
        return;
    }

    let idx = y * width + x;
    let channelSize = width * height;

    // For border, just copy original
    if (x < 1 || x >= (width - 1) || y < 1 || y >= (height - 1)) {
        store_pixel(idx, channelSize, load_pixel(idx, channelSize));
        return;
    }

    // Process each channel independently, so as to be identical to the cpu version in hotpixel.rs
    let newR = process_channel(idx, channelSize, x, y, 0);
    let newG = process_channel(idx, channelSize, x, y, 1);
    let newB = process_channel(idx, channelSize, x, y, 2);

    store_pixel(idx, channelSize, vec3<f32>(newR, newG, newB));
}
