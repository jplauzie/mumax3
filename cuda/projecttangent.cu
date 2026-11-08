#include "float3.h"

// v -= (v·m)/(m·m) m, and v = 0 where m = 0.
extern "C" __global__ void
projecttangent(float* __restrict__ vx, float* __restrict__ vy, float* __restrict__ vz,
               float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz, int N) {

    int i = ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {
        float3 m = make_float3(mx[i], my[i], mz[i]);
        float3 v = make_float3(vx[i], vy[i], vz[i]);
        float mm = dot(m, m);
        if (mm == 0.0f) {
            v = make_float3(0.0f, 0.0f, 0.0f);
        } else {
            v = v - (dot(v, m) / mm) * m;
        }
        vx[i] = v.x; vy[i] = v.y; vz[i] = v.z;
    }
}
