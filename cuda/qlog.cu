extern "C" __global__ void
unary_log(float* __restrict__ dst, float* __restrict__ a, int N) {
    int i = (blockIdx.y*gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if(i < N) { dst[i] = (a[i] > 0.0f) ? logf(a[i]) : 0.0f; }
}