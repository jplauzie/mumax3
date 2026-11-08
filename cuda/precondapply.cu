#include "precond.h"

// o = v + tau*K v   (o must not alias v)
extern "C" __global__ void
precondapply(float* __restrict__ ox, float* __restrict__ oy, float* __restrict__ oz,
             float* __restrict__ vx, float* __restrict__ vy, float* __restrict__ vz,
             float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
             float* __restrict__ aLUT2d, uint8_t* __restrict__ regions,
             float tau, float wx, float wy, float wz, int Nx, int Ny, int Nz, uint8_t PBC) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }
    int I = idx(ix, iy, iz);
    if (is0(make_float3(mx[I], my[I], mz[I]))) {
        ox[I] = 0.0f; oy[I] = 0.0f; oz[I] = 0.0f;
        return;
    }
    float D;
    float3 Kv = exch_lap(vx, vy, vz, mx, my, mz, aLUT2d, regions, wx, wy, wz,
                         ix, iy, iz, Nx, Ny, Nz, PBC, &D);
    float3 v0 = make_float3(vx[I], vy[I], vz[I]);
    float3 o = v0 + tau * Kv;
    ox[I] = o.x; oy[I] = o.y; oz[I] = o.z;
}
