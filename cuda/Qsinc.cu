// dst[i] = sinc(src[i]) = sin(x)/x, returns 1 if x == 0
extern "C" __global__ void
Qsinc(float* __restrict__ dst,
     float* __restrict__ src,
     int N) {

    int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = src[i];
        if (x != 0.0f) {
            dst[i] = sinf(x)/x;
        } else {
            dst[i] = 1.0f;
        }
    }
}