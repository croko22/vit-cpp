
#include <cmath>
#include <cuda_runtime.h>
#include "matrix_utils.cuh"

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