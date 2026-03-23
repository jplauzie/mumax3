extern "C" __global__ void
Qtan(float* __restrict__ dst,
    float* __restrict__ src,
    int N) {

    int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = src[i];
        float c = cosf(x);
        if (fabsf(c) > 1e-7f) {
            dst[i] = tanf(x);
        } else {
            dst[i] = 0.0f; // safe fallback
        }
    }
}