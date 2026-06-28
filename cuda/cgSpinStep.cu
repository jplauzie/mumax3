// cuda/cgSpinStep.cu
// CUDA kernel for geodesic spin rotation (FillBracket inner loop).
// Ported from OOMMF cgevolve.cc (M.J. Donahue, NIST).

#include <stdint.h>

extern "C" __global__ void
cgSpinStep(
        float* ox, float* oy, float* oz,             // output spin
        float* sx, float* sy, float* sz,             // best-pt spin
        float* dx, float* dy, float* dz,             // search direction
        float t0sq,        // bestpt.offset^2
        float dvec_scale,  // t_new - t_best
        int N)
{
    int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float Dx = dx[i], Dy = dy[i], Dz = dz[i];
    float dsq  = Dx*Dx + Dy*Dy + Dz*Dz;
    
    // If search direction is zero or non-magnetic, preserve original spin safely
    if (dsq < 1e-20f) {
        ox[i] = sx[i];
        oy[i] = sy[i];
        oz[i] = sz[i];
        return;
    }

    float mult = sqrtf(1.0f + t0sq * dsq);

    float nx = mult * sx[i] + dvec_scale * Dx;
    float ny = mult * sy[i] + dvec_scale * Dy;
    float nz = mult * sz[i] + dvec_scale * Dz;

    float normsq = nx*nx + ny*ny + nz*nz;
    if (normsq > 1e-20f) {
        float inv = rsqrtf(normsq);
        ox[i] = nx * inv;
        oy[i] = ny * inv;
        oz[i] = nz * inv;
    } else {
        // Fallback to original spin rather than flat zeroing out
        float s_norm = sx[i]*sx[i] + sy[i]*sy[i] + sz[i]*sz[i];
        if (s_norm > 0.0f) {
            float inv_s = rsqrtf(s_norm);
            ox[i] = sx[i] * inv_s;
            oy[i] = sy[i] * inv_s;
            oz[i] = sz[i] * inv_s;
        } else {
            ox[i] = 0.0f;
            oy[i] = 0.0f;
            oz[i] = 0.0f;
        }
    }
}