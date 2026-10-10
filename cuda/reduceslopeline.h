#ifndef REDUCESLOPELINE_H_
#define REDUCESLOPELINE_H_

#include <stdint.h>
#include "reduce.h"

// Must equal the launch block size (reducecfg.Block.X). Single source: reduce.h.
#define BLOCKDIM REDUCE_BLOCKSIZE

constexpr int warp_size = 32;

template <typename T>
__device__ __forceinline__ void kahanAdd(T &sum, T &c, T x) {
    T y = x - c;
    T t = sum + y;
    c = (t - sum) - y;
    sum = t;
}

template <typename T, int WIDTH = 32>
__device__ __forceinline__ T warpReduceSum(T val) {
    static_assert(WIDTH > 0 && (WIDTH & (WIDTH - 1)) == 0, "WIDTH must be a power of two");
    #pragma unroll
    for (int offset = WIDTH / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffffu, val, offset, WIDTH);
    return val;
}

template <typename T>
__device__ __forceinline__ T blockReduceSum(T val) {
    static_assert(BLOCKDIM % warp_size == 0, "assumes full warps");
    constexpr int NUM_WARPS = BLOCKDIM / warp_size;
    static_assert((NUM_WARPS & (NUM_WARPS - 1)) == 0, "BLOCKDIM/32 must be a power of two");
    __shared__ T sdata[NUM_WARPS];

    val = warpReduceSum<T, warp_size>(val);
    int lane = threadIdx.x % warp_size;
    int warpId = threadIdx.x / warp_size;
    if (lane == 0) sdata[warpId] = val;
    __syncthreads();

    if (warpId == 0) {
        T v = (lane < NUM_WARPS) ? sdata[lane] : T(0);
        val = warpReduceSum<T, NUM_WARPS>(v);
    }
    return val;
}

#endif
