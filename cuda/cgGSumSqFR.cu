// cuda/cgGSumSqFR.cu
// CUDA kernel for Fletcher-Reeves ||P g||^2 accumulation.
// Ported from OOMMF cgevolve.cc (M.J. Donahue, NIST).

#include <stdint.h>

__device__ static inline double warpSumD(double v) {
    for (int off = 16; off > 0; off >>= 1) {
        unsigned lo = __shfl_down_sync(0xffffffff, __double2loint(v), off);
        unsigned hi = __shfl_down_sync(0xffffffff, __double2hiint(v), off);
        v += __hiloint2double(hi, lo);
    }
    return v;
}

extern "C" __global__ void
cgGSumSqFR(
        float* g_out,
        float* mhx, float* mhy, float* mhz,
        float* msv,
        int N)
{
    __shared__ double smem[32];
    int nwarps = (blockDim.x + 31) >> 5;
    double val = 0.0;
    int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < N) {
        float mv = msv[i];
        if (mv != 0.0f) {
            double dmv = (double)mv;
            double mx = mhx[i], my = mhy[i], mz = mhz[i];
            val = (mx*mx + my*my + mz*mz) * (dmv*dmv);
        }
    }
    val = warpSumD(val);
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    double v = (threadIdx.x < nwarps) ? smem[threadIdx.x] : 0.0;
    if (wid == 0) v = warpSumD(v);
    if (threadIdx.x == 0) atomicAdd((double*)g_out, v);
}