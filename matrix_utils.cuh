#pragma once

__global__ void matmul(float *A, float *B, float *C, int N);
__global__ void row_softmax(float *mat, int N);
