#include <math.h>

// dst[i] = src1[i]^src2[i] with Mumax-style handling
// - Negative base + fractional exponent → move minus sign outside
// - 0 ^ negative exponent → return 0 (safe)
extern "C" __global__ void
Qpow(float* __restrict__ dst,
              float* __restrict__ src1,
              float* __restrict__ src2,
              int N) {

    int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < N) {
        float x = src1[i];
        float y = src2[i];

        // Guard: 0^negative → return 0
        if (x == 0.0f && y < 0.0f) {
            dst[i] = 0.0f;
            return;
        }

        // Check if exponent is effectively integer
        float y_floor = floorf(y);
        bool y_is_int = fabsf(y - y_floor) < 1e-6f;

        if (x < 0.0f && !y_is_int) {
            // Fractional exponent → move minus sign outside
            dst[i] = -1.0f * powf(fabsf(x), y);
        } else {
            // Integer exponent or positive base
            dst[i] = powf(x, y);
        }
    }
}