#include <stdint.h>
#include "float3.h"

extern "C" __global__ void
riemannian_displacement(
    float* outx, float* outy, float* outz,
    float* m0x, float* m0y, float* m0z,
    float* mx,  float* my,  float* mz,
    int N) {

    int i = (blockIdx.y * gridDim.x + blockIdx.x)
          * blockDim.x + threadIdx.x;

    if (i < N) {

        float ax = m0x[i];
        float ay = m0y[i];
        float az = m0z[i];

        float bx = mx[i];
        float by = my[i];
        float bz = mz[i];

        float c =
            ax * bx +
            ay * by +
            az * bz;

        c = fminf(1.0f, fmaxf(-1.0f, c));

        float theta = acosf(c);

        float factor;

        if (theta > 1e-6f) {
            float st = sinf(theta);
            factor = theta / st;
        } else {
            factor = 1.0f;
        }

        // Transported Log_{m0}(m), expressed in T_m S^2.
        //
        // s = theta/sin(theta) * (c*m - m0)
        //
        float sx = factor * (c * bx - ax);
        float sy = factor * (c * by - ay);
        float sz = factor * (c * bz - az);

        outx[i] = sx;
        outy[i] = sy;
        outz[i] = sz;
    }
}
