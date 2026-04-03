extern "C" __global__ void
pw_pow(float* __restrict__ dst, float* __restrict__ a, float* __restrict__ b, int N) {
    int i = (blockIdx.y*gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if(i < N) {
        float x = a[i];
        dst[i] = (x < 0.0f) ? -powf(-x, b[i]) : powf(x, b[i]);
    }
}