
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>
#include "matrix_utils.cuh"

__global__ void row_softmax(float *mat, int N)
{
    int row = blockIdx.x;
    if (row >= N)
        return;

    extern __shared__ float shared_data[];
    float *row_data = &mat[row * N];

    // Encontrar el máximo (reducción paralela)
    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        max_val = fmaxf(max_val, row_data[i]);
    shared_data[threadIdx.x] = max_val;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
    {
        if (threadIdx.x < offset)
            shared_data[threadIdx.x] = fmaxf(shared_data[threadIdx.x], shared_data[threadIdx.x + offset]);
        __syncthreads();
    }
    max_val = shared_data[0];

    // Calcular exponenciales y suma (reducción paralela)
    float sum = 0.0;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
    {
        row_data[i] = expf(row_data[i] - max_val);
        sum += row_data[i];
    }
    shared_data[threadIdx.x] = sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
    {
        if (threadIdx.x < offset)
            shared_data[threadIdx.x] += shared_data[threadIdx.x + offset];
        __syncthreads();
    }
    sum = shared_data[0];

    // Paso 3: Normalizar
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        row_data[i] /= sum;
}