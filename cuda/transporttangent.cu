#include <stdint.h>
#include "float3.h"

extern "C" __global__ void
transport_tangent(
    float* __restrict__ vx,
    float* __restrict__ vy,
    float* __restrict__ vz,
    float* __restrict__ m0x,
    float* __restrict__ m0y,
    float* __restrict__ m0z,
    float* __restrict__ mx,
    float* __restrict__ my,
    float* __restrict__ mz,
    int N)
{
    int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < N) {

        float3 v  = {vx[i], vy[i], vz[i]};
        float3 m0 = {m0x[i], m0y[i], m0z[i]};
        float3 m  = {mx[i],  my[i],  mz[i]};

        float denom = 1.0f + dot(m0, m);

        if (denom > 1e-12f) {

            float vm = dot(v, m);
            float coeff = vm / denom;

            float3 sum = m0 + m;

            v -= coeff * sum;
        }

        vx[i] = v.x;
        vy[i] = v.y;
        vz[i] = v.z;
    }
}