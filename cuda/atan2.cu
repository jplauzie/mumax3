extern "C" __global__ void
pw_atan2(float* __restrict__ dst, float* __restrict__ y, float* __restrict__ x, int N) {
    int i = (blockIdx.y*gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if(i < N) { dst[i] = atan2f(y[i], x[i]); }
}