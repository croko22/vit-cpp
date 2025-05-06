
#include <cmath>
#include <cuda_runtime.h>
#include "matmul_softmax.cuh"

__global__ void matmul(float *A, float *B, float *C, int N)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0;
    if (row < N && col < N)
    {
        for (int k = 0; k < N; ++k)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

__global__ void row_softmax(float *mat, int N)
{
    int row = blockIdx.x;
    if (row >= N)
        return;

    float max_val = mat[row * N];
    for (int i = 1; i < N; ++i)
        if (mat[row * N + i] > max_val)
            max_val = mat[row * N + i];

    float sum = 0.0;
    for (int i = 0; i < N; ++i)
    {
        mat[row * N + i] = expf(mat[row * N + i] - max_val);
        sum += mat[row * N + i];
    }

    for (int i = 0; i < N; ++i)
        mat[row * N + i] /= sum;
}
