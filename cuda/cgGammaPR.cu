// cuda/cgGammaPR.cu
// CUDA kernel for Polak-Ribière gamma numerator + update prev_mxHxm.
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

__device__ static inline void blockReduce2D(
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
cgGammaPR(
        float* g_gsq, float* g_gamma,
        float* curx, float* cury, float* curz,
        float* px,   float* py,   float* pz,
        float* msv,
        int N)
{
    __shared__ double smem[2 * 32];
    int nwarps = (blockDim.x + 31) >> 5;
    double* sgs = smem;
    double* sgm = smem + 32;
    double gs = 0.0, gm = 0.0;
    int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < N) {
        float mv = msv[i];
        if (mv != 0.0f) {
            double dmv2 = (double)mv * (double)mv;
            double cx = curx[i], cy = cury[i], cz = curz[i];
            double Px = px[i],   Py = py[i],   Pz = pz[i];
            gs = (cx*cx + cy*cy + cz*cz) * dmv2;
            gm = ((cx-Px)*cx + (cy-Py)*cy + (cz-Pz)*cz) * dmv2;
            px[i] = (float)cx;
            py[i] = (float)cy;
            pz[i] = (float)cz;
        } else {
            px[i] = 0.0f;
            py[i] = 0.0f;
            pz[i] = 0.0f;
        }
    }
    blockReduce2D(gs, gm, sgs, sgm, nwarps);
    if (threadIdx.x == 0) {
        atomicAdd((double*)g_gsq,   gs);
        atomicAdd((double*)g_gamma, gm);
    }
}