#include <stdint.h>
#include "reduceslopeline.h"

// Level 2: single block of BLOCKDIM threads, double accumulation. out[0] = sum of partial[0..np).
extern "C" __global__ void
reduceslopeline_combine(float* __restrict__ out, float* __restrict__ partial, int np) {
    if (blockDim.x != BLOCKDIM) __trap();   // launch config out of sync with reduce.h
    double acc = 0.0;
    for (int i = threadIdx.x; i < np; i += blockDim.x)
        acc += (double)partial[i];
    acc = blockReduceSum<double>(acc);
    if (threadIdx.x == 0) out[0] = (float)acc;
}
