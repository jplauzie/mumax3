//dst[i] = tan(a[i]), for poles at π/2 + nπ, returns 0
extern "C" __global__ void
Qtan(float* __restrict__ dst, float* __restrict__ src, int N) {
    int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < N) {
        dst[i] = tanf(src[i]);
    }
}