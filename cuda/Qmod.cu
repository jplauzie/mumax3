#include <math.h>

// dst[i] = fmod(src1[i], src2[i])
// Guard: division by zero → dst[i] = 0
extern "C" __global__ void
Qmod(float* __restrict__ dst,
              float* __restrict__ src1,
              float* __restrict__ src2,
              int N) {

    int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < N) {
        float a = src1[i];
        float b = src2[i];

        if (b != 0.0f) {
            dst[i] = fmodf(a, b);
        } else {
            dst[i] = 0.0f;
        }
    }
}