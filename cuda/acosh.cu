//dst[i] = acosh(a[i])
extern "C" __global__ void
unary_acosh(float *__restrict__ dst, float *__restrict__ a, int N)
{
    int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < N)
    {
        dst[i] = (a[i] >= 1.0f) ? acoshf(a[i]) : 0.0f;
    }
}