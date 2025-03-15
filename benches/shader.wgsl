@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, write> output : array<f32>;

@compute @workgroup_size(64, 1, 1)
fn doubleMe() {
    let index = gl_GlobalInvocationID.x;
    
    // Yep, the point is to DO NOTHING!
    output[index] = input[index];
}
