#include "precond.h"

// dst[i] = diagonal of the exchange stencil K (used for the Gershgorin bound).
extern "C" __global__ void
precondstiff(float* __restrict__ dst,
             float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
             float* __restrict__ aLUT2d, uint8_t* __restrict__ regions,
             float wx, float wy, float wz, int Nx, int Ny, int Nz, uint8_t PBC) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }
    int I = idx(ix, iy, iz);
    if (is0(make_float3(mx[I], my[I], mz[I]))) {
        dst[I] = 0.0f;
        return;
    }
    float D;
    exch_lap(mx, my, mz, mx, my, mz, aLUT2d, regions, wx, wy, wz, ix, iy, iz, Nx, Ny, Nz, PBC, &D);
    dst[I] = D;
}
