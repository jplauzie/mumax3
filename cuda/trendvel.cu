#include <stdint.h>

// End-point velocity of the Cayley retraction path used by the minimizer.
//
// The path is gamma(tau) = ((4 - tau^2 S) m0 + 4 tau u) / (4 + tau^2 S), S = |u|^2 (per cell),
// u = c*s tangent at m0. Its derivative at tau = 1 is
//     gamma'(1) = (4 (4 - S) u - 16 S m0) / (4 + S)^2.
// Used to integrate the gradient along the step with the trapezoid rule
// (initial velocity is u itself).
extern "C" __global__ void
trendvel(float* __restrict__ vx,  float* __restrict__ vy,  float* __restrict__ vz,
         float* __restrict__ m0x, float* __restrict__ m0y, float* __restrict__ m0z,
         float* __restrict__ sx,  float* __restrict__ sy,  float* __restrict__ sz,
         float c, int N) {

    int i = ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {
        float ux = c * sx[i];
        float uy = c * sy[i];
        float uz = c * sz[i];

        float S   = ux*ux + uy*uy + uz*uz;
        float den = (4.0f + S) * (4.0f + S);
        float a   = 4.0f * (4.0f - S) / den;
        float b   = 16.0f * S / den;

        vx[i] = a*ux - b*m0x[i];
        vy[i] = a*uy - b*m0y[i];
        vz[i] = a*uz - b*m0z[i];
    }
}
