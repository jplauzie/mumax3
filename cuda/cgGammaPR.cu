// cuda/cgGammaPR.cu
// CUDA kernel for Polak-Ribière gamma numerator + update prev_mxHxm.
// Ported from OOMMF cgevolve.cc (M.J. Donahue, NIST).

#include <stdint.h>

__device__ static inline float warpSum(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

__device__ static inline void blockReduce2(
        float& a, float& b,
        float* sa, float* sb, int nwarps)
{
    a = warpSum(a);
    b = warpSum(b);

    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    if (lane == 0) { sa[wid] = a; sb[wid] = b; }
    __syncthreads();

    float av = (threadIdx.x < nwarps) ? sa[threadIdx.x] : 0.f;
    float bv = (threadIdx.x < nwarps) ? sb[threadIdx.x] : 0.f;
    if (wid == 0) { av = warpSum(av); bv = warpSum(bv); }

    a = av; b = bv;
}

extern "C" __global__ void
cgGammaPR(
        float* g_gsq, float* g_gamma,
         float* curx,  float* cury,  float* curz, // current mxHxm
        float* px, float* py, float* pz,          // prev mxHxm (in/out)
         float* msv,
        int N)
{
    __shared__ float smem[2 * 32];
    int nwarps = (blockDim.x + 31) >> 5;
    float* sgs = smem;
    float* sgm = smem + 32;

    float gs = 0.f, gm = 0.f;
    int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < N) {
        float mv  = msv[i];
        
        // Insulate against 0 * NaN = NaN
        if (mv != 0.0f) {
            float mv2 = mv * mv;
            float cx = curx[i], cy = cury[i], cz = curz[i];
            float Px = px[i],   Py = py[i],   Pz = pz[i];

            gs = (cx*cx + cy*cy + cz*cz) * mv2;
            gm = ((cx-Px)*cx + (cy-Py)*cy + (cz-Pz)*cz) * mv2;

            px[i] = cx;  py[i] = cy;  pz[i] = cz;
        } else {
            // Actively sanitize the previous direction buffers in non-magnetic regions
            px[i] = 0.0f;
            py[i] = 0.0f;
            pz[i] = 0.0f;
        }
    }

    blockReduce2(gs, gm, sgs, sgm, nwarps);

    if (threadIdx.x == 0) {
        atomicAdd(g_gsq,   gs);
        atomicAdd(g_gamma, gm);
    }
}