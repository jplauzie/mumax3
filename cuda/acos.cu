//dst[i] = acos(a[i])

extern "C" __global__ void
unary_acos(float *__restrict__ dst, float *__restrict__ a, int N)
{
    int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < N)
    {
        dst[i] =  acosf(a[i]);
    }
}