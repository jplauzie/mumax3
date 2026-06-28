// cuda/cgbuildmsv.cu
// Builds the per-cell Ms*Volume weight buffer used by the CG kernels.
//
// mumax3's MSlice convention:
//   - For spatially varying Ms: msArr[i] holds per-cell values, actual = msArr[i]*msMul
//   - For uniform Ms: msArr is NULL (DevPtr returns nil), actual = msMul
//
// We bake in cell volume at the same time: dst[i] = actual_Ms[i] * scale

extern "C" __global__ void
cgBuildMsV(float* dst, float* msArr, float msMul, float scale, int N) {
    int i = ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float ms = (msArr == 0) ? msMul : msArr[i] * msMul;
    dst[i] = ms * scale;
}