#include <stdint.h>
#include "float3.h"

// Steepest descent update using the exact rotation (geodesic) instead of the Cayley transform.
// t must be tangent to m0, which holds for the torque here.
extern "C" __global__ void
minimizeexp(float* __restrict__ mx,  float* __restrict__ my,  float* __restrict__ mz,
            float* __restrict__ m0x, float* __restrict__ m0y, float* __restrict__ m0z,
            float* __restrict__ tx,  float* __restrict__ ty,  float* __restrict__ tz,
            float dt, int N) {

    int i = ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {

        float3 m0 = {m0x[i], m0y[i], m0z[i]};
        float3 t  = {tx[i], ty[i], tz[i]};

        float phi  = dt * sqrtf(dot(t, t));   // signed rotation angle
        float c    = cosf(phi);
        float sinc = (fabsf(phi) > 1e-3f) ? sinf(phi) / phi : 1.0f - phi*phi/6.0f;

        // cos(phi)*m0 + sin(phi)*t/|t|, written as dt*sinc*t so that |t| = 0 needs no special case
        float3 result = c * m0 + (dt * sinc) * t;

        mx[i] = result.x;
        my[i] = result.y;
        mz[i] = result.z;
    }
}
