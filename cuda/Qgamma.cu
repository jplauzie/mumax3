// dst[i] = gamma(src[i]) = tgamma(src[i]), returns 0 for x <= 0
extern "C" __global__ void
Qgamma(float* __restrict__ dst,
      float* __restrict__ src,
      int N) {

    int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < N) {
        float x = src[i];
        if (x > 0.0f) {
            dst[i] = tgammaf(x);
        } else {
            dst[i] = 0.0f;
        }
    }
}