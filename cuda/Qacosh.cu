// dst[i] = acosh(src[i]), returns 0 for x < 1
extern "C" __global__ void
Qacosh(float* __restrict__ dst,
      float* __restrict__ src,
      int N) {

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