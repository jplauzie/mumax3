// dst[i] = log(a[i]), returns 0 for non-positive input
extern "C" __global__ void
Qlog(float* __restrict__ dst, float* __restrict__ src, int N) {
    int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < N) {
        if (src[i] > 0.0f) {
            dst[i] = logf(src[i]);
        } else {
            dst[i] = 0.0f;
        }
    }
}