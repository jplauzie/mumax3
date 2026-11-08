#ifndef _PRECOND_H_
#define _PRECOND_H_

#include <stdint.h>
#include "exchange.h"
#include "float3.h"
#include "stencil.h"

// Handles one neighbor. Skips self (clamped/wrapped boundary) and empty (m==0) cells.
#define PRECOND_NEIGH(IDX, W)                                           \
    {                                                                   \
        int i_ = (IDX);                                                 \
        float3 m_ = make_float3(mx[i_], my[i_], mz[i_]);                \
        if (i_ != I && !is0(m_)) {                                      \
            float c_ = (W) * aLUT2d[symidx(r0, regions[i_])];           \
            acc += c_ * (v0 - make_float3(vx[i_], vy[i_], vz[i_]));     \
            D += c_;                                                    \
        }                                                               \
    }

// Returns (K v) at (ix,iy,iz), K = exchange-stencil graph Laplacian. *diag = stencil diagonal.
// Parameter names Nx,Ny,Nz,PBC must stay as-is: the stencil.h macros use them.
static __device__ __forceinline__ float3
exch_lap(const float* vx, const float* vy, const float* vz,
         const float* mx, const float* my, const float* mz,
         const float* aLUT2d, const uint8_t* regions,
         float wx, float wy, float wz,
         int ix, int iy, int iz, int Nx, int Ny, int Nz, uint8_t PBC,
         float* diag) {
    int I = idx(ix, iy, iz);
    uint8_t r0 = regions[I];
    float3 v0 = make_float3(vx[I], vy[I], vz[I]);
    float3 acc = make_float3(0.0f, 0.0f, 0.0f);
    float D = 0.0f;

    PRECOND_NEIGH(idx(lclampx(ix-1), iy, iz), wx)
    PRECOND_NEIGH(idx(hclampx(ix+1), iy, iz), wx)
    PRECOND_NEIGH(idx(ix, lclampy(iy-1), iz), wy)
    PRECOND_NEIGH(idx(ix, hclampy(iy+1), iz), wy)
    if (Nz != 1) {
        PRECOND_NEIGH(idx(ix, iy, lclampz(iz-1)), wz)
        PRECOND_NEIGH(idx(ix, iy, hclampz(iz+1)), wz)
    }
    *diag = D;
    return acc;
}

#endif
