// cuda/cgUpdateDir.cu
// CUDA kernel for CG direction update with tangent projection + stats.
// Ported from OOMMF cgevolve.cc (M.J. Donahue, NIST).

//VERY SKETCHY FLOAT*,to make cuda2go work with double* (for ep and gradsumsq) without changing the interface.

#include <stdint.h>

__device__ static inline float warpSum(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

__device__ static inline float warpMax(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
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

__device__ static inline void blockReduce4_mixed(
        float& mm, float& ns, double& ep, double& gs,
        float* smm, float* sns, double* sep, double* sgs, int nwarps)
{
    mm = warpMax(mm);
    ns = warpSum(ns);
    ep = warpSumD(ep);
    gs = warpSumD(gs);
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    if (lane == 0) {
        smm[wid] = mm; sns[wid] = ns;
        sep[wid] = ep; sgs[wid] = gs;
    }
    __syncthreads();
    float mmv = (threadIdx.x < nwarps) ? smm[threadIdx.x] : 0.f;
    float nsv = (threadIdx.x < nwarps) ? sns[threadIdx.x] : 0.f;
    double epv = (threadIdx.x < nwarps) ? sep[threadIdx.x] : 0.0;
    double gsv = (threadIdx.x < nwarps) ? sgs[threadIdx.x] : 0.0;
    if (wid == 0) {
        mmv = warpMax(mmv);
        nsv = warpSum(nsv);
        epv = warpSumD(epv);
        gsv = warpSumD(gsv);
    }
    mm = mmv; ns = nsv; ep = epv; gs = gsv;
}

extern "C" __global__ void
cgUpdateDir(
        float* dirx, float* diry, float* dirz,
        float* mhx,  float* mhy,  float* mhz,
        float* spx,  float* spy,  float* spz,
        float* msv,
        float gamma,
        float*  g_maxmagsq, float*  g_normsumsq,
        float*  g_ep_raw,   float*  g_gradsumsq_raw,  // ← still float* for cuda2go
        int N)
{
    // smem layout: 32 floats for mm, 32 floats for ns,
    //              32 doubles for ep, 32 doubles for gs
    __shared__ float  smm[32];
    __shared__ float  sns[32];
    __shared__ double sep[32];
    __shared__ double sgs[32];
    int nwarps = (blockDim.x + 31) >> 5;

    float mm = 0.f, ns = 0.f;
    double ep = 0.0, gs = 0.0;

    int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < N) {
        float mv = msv[i];
        float tx = 0.f, ty = 0.f, tz = 0.f;
        if (mv != 0.0f) {
            float tqx = mhx[i], tqy = mhy[i], tqz = mhz[i];
            tx = mv * tqx;
            ty = mv * tqy;
            tz = mv * tqz;
            if (gamma != 0.0f) {
                tx += gamma * dirx[i];
                ty += gamma * diry[i];
                tz += gamma * dirz[i];
            }
            float sx = spx[i], sy = spy[i], sz = spz[i];
            float dot = tx*sx + ty*sy + tz*sz;
            tx -= dot*sx;
            ty -= dot*sy;
            tz -= dot*sz;
            mm = tx*tx + ty*ty + tz*tz;
            ns = mm;
            // ep and gs in double to avoid catastrophic cancellation
            double dmv  = (double)mv;
            double dtqx = (double)tqx, dtqy = (double)tqy, dtqz = (double)tqz;
            double dtx  = (double)tx,  dty  = (double)ty,  dtz  = (double)tz;
            ep = (dtx*dtqx + dty*dtqy + dtz*dtqz) * dmv;
            gs = (dtqx*dtqx + dtqy*dtqy + dtqz*dtqz) * (dmv*dmv);
        }
        dirx[i] = tx;
        diry[i] = ty;
        dirz[i] = tz;
    }
    blockReduce4_mixed(mm, ns, ep, gs, smm, sns, sep, sgs, nwarps);
    if (threadIdx.x == 0) {
        atomicMax((unsigned int*)g_maxmagsq, __float_as_uint(fabsf(mm)));
        atomicAdd(g_normsumsq, ns);
        // Reinterpret float* as double* for the atomic
        atomicAdd((double*)g_ep_raw,        ep);
        atomicAdd((double*)g_gradsumsq_raw, gs);
    }
}