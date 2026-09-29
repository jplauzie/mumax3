#include <stdint.h>
#include "float3.h"

// Riemannian secant pair, per cell, both expressed in the tangent plane at the OLD m.
extern "C" __global__ void
riemannsy(float* __restrict__ sx,  float* __restrict__ sy,  float* __restrict__ sz,
          float* __restrict__ yx,  float* __restrict__ yy,  float* __restrict__ yz,
          float* __restrict__ m0x, float* __restrict__ m0y, float* __restrict__ m0z,
          float* __restrict__ m1x, float* __restrict__ m1y, float* __restrict__ m1z,
          float* __restrict__ t0x, float* __restrict__ t0y, float* __restrict__ t0z,
          float* __restrict__ t1x, float* __restrict__ t1y, float* __restrict__ t1z,
          int N) {

    int i = ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {
        float3 a  = {m0x[i], m0y[i], m0z[i]};  // old m
        float3 b  = {m1x[i], m1y[i], m1z[i]};  // new m
        float3 ta = {t0x[i], t0y[i], t0z[i]};  // old torque
        float3 tb = {t1x[i], t1y[i], t1z[i]};  // new torque

        float3 d  = b - a;
        float  e  = dot(d, a);                 // = cos(phi) - 1, accurate for small steps
        float  cs = 1.0f + e;
        float3 c  = cross(a, b);
        float  sn = sqrtf(dot(c, c));

        float scale = 1.0f;
        if (sn > 1e-12f) {
            scale = atan2f(sn, cs) / sn;
        }
        float3 s  = scale * (d - e * a);       // tangent projection of d, no cancellation

        float3 tr = cs * tb - cross(c, tb) + (dot(c, tb) / (2.0f + e)) * c;
        float3 y  = ta - tr;

        sx[i] = s.x; sy[i] = s.y; sz[i] = s.z;
        yx[i] = y.x; yy[i] = y.y; yz[i] = y.z;
    }
}
