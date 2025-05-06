
#include <cuda_runtime.h>
#include "attention.cuh"
#include "matrix_utils.cuh"

void run_attention(float *h_Q, float *h_K, float *h_V, float *h_output, int N)
{
    float *d_Q, *d_K, *d_V, *d_score, *d_out;
    size_t size = N * N * sizeof(float);

    cudaMalloc(&d_Q, size);
    cudaMalloc(&d_K, size);
    cudaMalloc(&d_V, size);
    cudaMalloc(&d_score, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_Q, h_Q, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (N + 15) / 16);

    matmul<<<blocks, threads>>>(d_Q, d_K, d_score, N);
    row_softmax<<<N, 1>>>(d_score, N);
    matmul<<<blocks, threads>>>(d_score, d_V, d_out, N);

    cudaMemcpy(h_output, d_out, size, cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_score);
    cudaFree(d_out);
}
