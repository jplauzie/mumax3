// dst[i] = gamma(a[i])
extern "C" __global__ void
unary_gamma(float *__restrict__ dst, float *__restrict__ a, int N)
{
    int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < N)
    {
        float x = a[i];
        dst[i] = (x > 0.0f || (x != floorf(x))) ? tgammaf(x) : 0.0f;
    }
}