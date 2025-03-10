// hot_pixel_correction.wgsl
struct Params {
    width: u32,
    height: u32,
    window_size: i32,
    threshold: f32,
};

@group(0) @binding(0)
var<storage, read> input: array<f32>;
// Our input array is flattened as [R[...], G[...], B[...]]
@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size(16,16,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = params.width;
    let height = params.height;
    let channelSize = width * height;
    if (gid.x >= width || gid.y >= height) {
        return;
    }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let idx = y * i32(width) + i32(gid.x);

    // For border pixels, copy all channels.
    if (x < 1 || x >= i32(width)-1 || y < 1 || y >= i32(height)-1) {
        output[idx] = input[idx];
        output[idx + i32(channelSize)] = input[idx + i32(channelSize)];
        output[idx + i32(2u * channelSize)] = input[idx + i32(2u * channelSize)];
        return;
    }

    let half = params.window_size / 2;
    var sumR: f32 = 0.0;
    var sumG: f32 = 0.0;
    var sumB: f32 = 0.0;
    var count: i32 = 0;
    // Accumulate sums for each channel.
    for (var j: i32 = -half; j <= half; j = j + 1) {
        for (var i: i32 = -half; i <= half; i = i + 1) {
            let nx = x + i;
            let ny = y + j;
            if (nx >= 0 && nx < i32(width) && ny >= 0 && ny < i32(height)) {
                let nidx = ny * i32(width) + nx;
                sumR = sumR + input[nidx];
                sumG = sumG + input[nidx + i32(channelSize)];
                sumB = sumB + input[nidx + i32(2u * channelSize)];
                count = count + 1;
            }
        }
    }
    let meanR = sumR / f32(count);
    let meanG = sumG / f32(count);
    let meanB = sumB / f32(count);

    // Accumulate squared differences.
    var sumSqR: f32 = 0.0;
    var sumSqG: f32 = 0.0;
    var sumSqB: f32 = 0.0;
    for (var j: i32 = -half; j <= half; j = j + 1) {
        for (var i: i32 = -half; i <= half; i = i + 1) {
            let nx = x + i;
            let ny = y + j;
            if (nx >= 0 && nx < i32(width) && ny >= 0 && ny < i32(height)) {
                let nidx = ny * i32(width) + nx;
                let diffR = input[nidx] - meanR;
                let diffG = input[nidx + i32(channelSize)] - meanG;
                let diffB = input[nidx + i32(2u * channelSize)] - meanB;
                sumSqR = sumSqR + diffR * diffR;
                sumSqG = sumSqG + diffG * diffG;
                sumSqB = sumSqB + diffB * diffB;
            }
        }
    }
    let stddevR = sqrt(sumSqR / f32(count));
    let stddevG = sqrt(sumSqG / f32(count));
    let stddevB = sqrt(sumSqB / f32(count));

    // Get current pixel values.
    let currR = input[idx];
    let currG = input[idx + i32(channelSize)];
    let currB = input[idx + i32(2u * channelSize)];


    // Compute z-scores (guard against division by zero) using select:
    let zR = select(0.0, abs(currR - meanR) / stddevR, stddevR > 0.0);
    let zG = select(0.0, abs(currG - meanG) / stddevG, stddevG > 0.0);
    let zB = select(0.0, abs(currB - meanB) / stddevB, stddevB > 0.0);


    var outR = currR;
    var outG = currG;
    var outB = currB;
    if (zR > params.threshold) { outR = meanR; }
    if (zG > params.threshold) { outG = meanG; }
    if (zB > params.threshold) { outB = meanB; }

    output[idx] = outR;
    output[idx + i32(channelSize)] = outG;
    output[idx + i32(2u * channelSize)] = outB;
}