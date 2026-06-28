#include <stdint.h>
#include "float3.h"

// Project v onto the tangent plane of m:
// v <- v - (v·m)m
extern "C" __global__ void
project_tangent(
    float* __restrict__ vx,
    float* __restrict__ vy,
    float* __restrict__ vz,
    float* __restrict__ mx,
    float* __restrict__ my,
    float* __restrict__ mz,
    int N)
{
    int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < N) {

        float3 v = {vx[i], vy[i], vz[i]};
        float3 m = {mx[i], my[i], mz[i]};

        float vm = dot(v, m);

        v -= vm * m;

        vx[i] = v.x;
        vy[i] = v.y;
        vz[i] = v.z;
    }
}