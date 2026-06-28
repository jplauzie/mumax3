// cuda/cgGSumSqFR.cu
// CUDA kernel for Fletcher-Reeves ||P g||^2 accumulation.
// Ported from OOMMF cgevolve.cc (M.J. Donahue, NIST).

#include <stdint.h>

__device__ static inline float warpSum(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

extern "C" __global__ void
cgGSumSqFR(
        float* g_out,
         float* mhx,  float* mhy,  float* mhz,
         float* msv,
        int N)
{
    __shared__ float smem[32];       // 1 output × 32 warps max
    int nwarps = (blockDim.x + 31) >> 5;

    float val = 0.f;
    int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < N) {
        float mv = msv[i];
        
        // Insulate against 0 * NaN = NaN
        if (mv != 0.0f) {
            float mx = mhx[i], my = mhy[i], mz = mhz[i];
            val = (mx*mx + my*my + mz*mz) * (mv*mv);
        }
    }

    val = warpSum(val);
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    if (lane == 0) smem[wid] = val;
    __syncthreads();

    float v = (threadIdx.x < nwarps) ? smem[threadIdx.x] : 0.f;
    if (wid == 0) v = warpSum(v);
    if (threadIdx.x == 0) atomicAdd(g_out, v);
}