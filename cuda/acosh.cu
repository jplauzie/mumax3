//dst[i] = acosh(a[i]), returns 0 for a[i] outside of domain [1, inf)

extern "C" __global__ void
Qacosh(float* __restrict__ dst, float* __restrict__ src, int N) {
    int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < N) {
        float x = src[i];
        if (x >= 1.0f) {
            dst[i] = acoshf(x);
        } else {
            dst[i] = 0.0f;
        }
    }
}