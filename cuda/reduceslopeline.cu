#include <stdint.h>
#include "reduceslopeline.h"

#ifndef BLOCKDIM
#define BLOCKDIM 512 // MUST equal reducecfg.Block.X; multiple of 32, at most 1024
#endif

// Level 1: partial[blockIdx.x] = sum over this block's cells of (g.s)/|wa + a*s|.
extern "C" __global__ void
reduceslopeline_partial(float* __restrict__ gx,  float* __restrict__ gy,  float* __restrict__ gz,
                        float* __restrict__ wax, float* __restrict__ way, float* __restrict__ waz,
                        float* __restrict__ sx,  float* __restrict__ sy,  float* __restrict__ sz,
                        float* __restrict__ partial, float a, int n) {

    float sum = 0.0f, comp = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float xx = wax[i] + a * sx[i];
        float xy = way[i] + a * sy[i];
        float xz = waz[i] + a * sz[i];
        float n2 = xx * xx + xy * xy + xz * xz;
        if (n2 > 0.0f) { // cells outside the geometry have x = 0
            float gs = gx[i] * sx[i] + gy[i] * sy[i] + gz[i] * sz[i];
            kahanAdd(sum, comp, gs * rsqrtf(n2));
        }
    }

    sum = blockReduceSum<float>(sum);
    if (threadIdx.x == 0) partial[blockIdx.x] = sum;
}
