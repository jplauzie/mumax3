// dst[i] = exp(src[i]);
extern "C" __global__ void
Qexp(float* __restrict__ dst,
    float* __restrict__ src,
    int N) {

    int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < N) {
        dst[i] = expf(src[i]);
    }
}