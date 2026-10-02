#include <stdint.h>
#include "reduceslopeline.h"

#ifndef BLOCKDIM
#define BLOCKDIM 512 // MUST equal reducecfg.Block.X; multiple of 32, at most 1024
#endif



// Level 2: single block of BLOCKDIM threads, double accumulation. out[0] = sum of partial[0..np).
extern "C" __global__ void
reduceslopeline_combine(float* __restrict__ out, float* __restrict__ partial, int np) {
    double acc = 0.0;
    for (int i = threadIdx.x; i < np; i += blockDim.x)
        acc += (double)partial[i];
    acc = blockReduceSum<double>(acc);
    if (threadIdx.x == 0) out[0] = (float)acc;
}
