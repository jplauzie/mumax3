// cuda/cgEpAndGradSq.cu
// CUDA kernel for weighted reduction for line derivative + grad norm.
// Ported from OOMMF cgevolve.cc (M.J. Donahue, NIST).

#include <stdint.h>

__device__ static inline float warpSum(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

__device__ static inline double warpSumD(double v) {
    // Shuffle as two 32-bit halves
    for (int off = 16; off > 0; off >>= 1) {
        unsigned lo = __shfl_down_sync(0xffffffff, __double2loint(v), off);
        unsigned hi = __shfl_down_sync(0xffffffff, __double2hiint(v), off);
        v += __hiloint2double(hi, lo);
    }
    return v;
}

__device__ static inline void blockReduce2_df(
        double& a, double& b,
        double* sa, double* sb, int nwarps)
{
    a = warpSumD(a);
    b = warpSumD(b);
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    if (lane == 0) { sa[wid] = a; sb[wid] = b; }
    __syncthreads();
    double av = (threadIdx.x < nwarps) ? sa[threadIdx.x] : 0.0;
    double bv = (threadIdx.x < nwarps) ? sb[threadIdx.x] : 0.0;
    if (wid == 0) { av = warpSumD(av); bv = warpSumD(bv); }
    a = av; b = bv;
}

extern "C" __global__ void
cgEpAndGradSq(
        float* g_ep, float* g_gsq,
        float* mhx,  float* mhy,  float* mhz,
        float* dirx, float* diry, float* dirz,
        float* ms,
        float tsq, int N)
{
    __shared__ double smem[2 * 32];
    int nwarps = (blockDim.x + 31) >> 5;
    double* sep  = smem;
    double* sgsq = smem + 32;

    double ep = 0.0, gsq = 0.0;
    int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < N) {
        float s = ms[i];
        if (s != 0.0f) {
            double Dx = dirx[i], Dy = diry[i], Dz = dirz[i];
            double Mx = mhx[i],  My = mhy[i],  Mz = mhz[i];
            double dsq = Dx*Dx + Dy*Dy + Dz*Dz;
            double sd = (double)s / sqrt(1.0 + (double)tsq * dsq);
            ep  += (Mx*Dx + My*Dy + Mz*Dz) * sd;
            gsq += (Mx*Mx + My*My + Mz*Mz) * (sd * sd);
        }
    }
    blockReduce2_df(ep, gsq, sep, sgsq, nwarps);
    if (threadIdx.x == 0) {
        atomicAdd(g_ep,  (float)ep);
        atomicAdd(g_gsq, (float)gsq);
    }
}