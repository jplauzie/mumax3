#include <stdint.h>
#include "float3.h"

// Parallel transport of a tangent vector k0 from the tangent plane
// at m0 to the tangent plane at m.
//
// kT = k0 - [(k0 · m) / (1 + m0 · m)] (m0 + m)
//
// m0 and m should both be normalized.
// k0 should be tangent to m0.

extern "C" __global__ void
parallel_transport(
    float* outx, float* outy, float* outz,
    float* m0x,  float* m0y,  float* m0z,
    float* mx,   float* my,   float* mz,
    float* k0x,  float* k0y,  float* k0z,
    int N) {

    int i = (blockIdx.y * gridDim.x + blockIdx.x)
          * blockDim.x + threadIdx.x;

    if (i < N) {

        float3 m0 = {m0x[i], m0y[i], m0z[i]};
        float3 m  = {mx[i],  my[i],  mz[i]};
        float3 k0 = {k0x[i], k0y[i], k0z[i]};

        // m0 · m
        float mm =
            m0.x * m.x +
            m0.y * m.y +
            m0.z * m.z;

        // k0 · m
        float km =
            k0.x * m.x +
            k0.y * m.y +
            k0.z * m.z;

        float denom = 1.0f + mm;

        float3 kT;

        if (denom > 1e-7f) {

            float coeff = km / denom;

            // kT = k0 - coeff * (m0 + m)
            kT.x = k0.x - coeff * (m0.x + m.x);
            kT.y = k0.y - coeff * (m0.y + m.y);
            kT.z = k0.z - coeff * (m0.z + m.z);

        } else {

            // m0 and m are nearly antiparallel.
            // Fall back to projection onto the new tangent plane.

            kT.x = k0.x - km * m.x;
            kT.y = k0.y - km * m.y;
            kT.z = k0.z - km * m.z;
        }

        outx[i] = kT.x;
        outy[i] = kT.y;
        outz[i] = kT.z;
    }
}
