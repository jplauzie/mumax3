// dst[i] = heaviside(src[i])
// returns 0 for x < 0, 0.5 for x == 0, 1 for x > 0
extern "C" __global__ void
Qheaviside(float* __restrict__ dst,
          float* __restrict__ src,
          int N) {

    int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < N) {
        float x = src[i];

        if (x > 0.0f) {
            dst[i] = 1.0f;
        } else if (x < 0.0f) {
            dst[i] = 0.0f;
        } else {
            // x == 0
            dst[i] = 0.5f;
        }
    }
}