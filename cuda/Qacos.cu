// dst[i] = acos(src[i]), returns 0 outside [-1, 1]
extern "C" __global__ void
Qacos(float* __restrict__ dst,
     float* __restrict__ src,
     int N) {

    int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < N) {
        float x = src[i];
        if (x >= -1.0f && x <= 1.0f) {
            dst[i] = acosf(x);
        } else {
            dst[i] = 0.0f;
        }
    }
}