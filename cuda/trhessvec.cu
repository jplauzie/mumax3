#include <stdint.h>

// Riemannian Hessian-vector product from a finite difference of the torque.
//
// t0 = torque at m, t1 = torque at R_m(eps*d). The torque is t = P_m B_eff, i.e. the
// negative Riemannian gradient. d/deps t, projected back on the tangent space at m,
// equals -Hess[d], so
//     h = Hess[d] ~= -P_m[(t1 - t0)/eps],   P_m w = w - (m.w) m.
// Pass invEps = 1/eps.
extern "C" __global__ void
trhessvec(float* __restrict__ hx,  float* __restrict__ hy,  float* __restrict__ hz,
          float* __restrict__ t1x, float* __restrict__ t1y, float* __restrict__ t1z,
          float* __restrict__ t0x, float* __restrict__ t0y, float* __restrict__ t0z,
          float* __restrict__ mx,  float* __restrict__ my,  float* __restrict__ mz,
          float invEps, int N) {

    int i = ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {
        float wx = (t1x[i] - t0x[i]) * invEps;
        float wy = (t1y[i] - t0y[i]) * invEps;
        float wz = (t1z[i] - t0z[i]) * invEps;

        float mmx = mx[i];
        float mmy = my[i];
        float mmz = mz[i];

        float mw = mmx*wx + mmy*wy + mmz*wz;

        hx[i] = mw*mmx - wx;
        hy[i] = mw*mmy - wy;
        hz[i] = mw*mmz - wz;
    }
}
