#include <stdint.h>
#include "float3.h"

extern "C" __global__ void
precond_step(
    float* __restrict__ px,   float* __restrict__ py,   float* __restrict__ pz,
    float* __restrict__ avg,
    float* __restrict__ kx,   float* __restrict__ ky,   float* __restrict__ kz,
    float decay, float eps, int N)
{
    int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < N) {
        float om = 1.0f - decay;
        float k2 = kx[i]*kx[i] + ky[i]*ky[i] + kz[i]*kz[i];
        float a  = decay * avg[i] + om * k2;
        avg[i] = a;
        float scale = 1.0f / (sqrtf(a) + eps);
        px[i] = kx[i] * scale;
        py[i] = ky[i] * scale;
        pz[i] = kz[i] * scale;
    }
}
