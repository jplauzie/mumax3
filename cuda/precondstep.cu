#include "precond.h"

// One fused Chebyshev iteration for A z = r, A = I + tau*K:
//   res -= A d_old;  d_new = c1*d_old + c2*res;  z += d_new
// d_new and d_old must be different buffers (neighbors read d_old).
extern "C" __global__ void
precondstep(float* __restrict__ dnx, float* __restrict__ dny, float* __restrict__ dnz,
            float* __restrict__ dox, float* __restrict__ doy, float* __restrict__ doz,
            float* __restrict__ rx,  float* __restrict__ ry,  float* __restrict__ rz,
            float* __restrict__ zx,  float* __restrict__ zy,  float* __restrict__ zz,
            float* __restrict__ mx,  float* __restrict__ my,  float* __restrict__ mz,
            float* __restrict__ aLUT2d, uint8_t* __restrict__ regions,
            float tau, float c1, float c2, float wx, float wy, float wz,
            int Nx, int Ny, int Nz, uint8_t PBC) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }
    int I = idx(ix, iy, iz);
    if (is0(make_float3(mx[I], my[I], mz[I]))) {
        dnx[I] = 0.0f; dny[I] = 0.0f; dnz[I] = 0.0f;
        return;
    }
    float D;
    float3 Kd = exch_lap(dox, doy, doz, mx, my, mz, aLUT2d, regions, wx, wy, wz,
                         ix, iy, iz, Nx, Ny, Nz, PBC, &D);
    float3 d0  = make_float3(dox[I], doy[I], doz[I]);
    float3 res = make_float3(rx[I], ry[I], rz[I]) - (d0 + tau * Kd);
    float3 dn  = c1 * d0 + c2 * res;
    float3 z   = make_float3(zx[I], zy[I], zz[I]) + dn;

    rx[I] = res.x; ry[I] = res.y; rz[I] = res.z;
    dnx[I] = dn.x; dny[I] = dn.y; dnz[I] = dn.z;
    zx[I] = z.x;   zy[I] = z.y;   zz[I] = z.z;
}
