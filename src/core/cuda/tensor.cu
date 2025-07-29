#include "../../../include/core/cuda/tensor.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/transform.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>

// Inicializar variables estáticas
cublasHandle_t TensorCuda::cublas_handle;
bool TensorCuda::cublas_initialized = false;

// --- Kernels CUDA ---

__global__ void fill_kernel(float* data, int size, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = value;
    }
}

__global__ void add_kernel(const float* a, const float* b, float* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] + b[idx];
    }
}

__global__ void subtract_kernel(const float* a, const float* b, float* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] - b[idx];
    }
}

__global__ void scalar_multiply_kernel(const float* a, float scalar, float* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] * scalar;
    }
}

__global__ void transpose_2d_kernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
}

__global__ void setup_random_states(curandState* states, unsigned long seed, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void xavier_init_kernel(float* data, curandState* states, int size, float std_dev) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = curand_normal(&states[idx]) * std_dev;
    }
}

__global__ void he_init_kernel(float* data, curandState* states, int size, float std_dev) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = curand_normal(&states[idx]) * std_dev;
    }
}

// --- Implementación de TensorCuda ---

void TensorCuda::check_cuda_error(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error en " << file << ":" << line << " - " 
                  << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error("Error de CUDA");
    }
}

void TensorCuda::init_cublas() {
    if (!cublas_initialized) {
        cublasStatus_t status = cublasCreate(&cublas_handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Error al inicializar cuBLAS");
        }
        cublas_initialized = true;
    }
}

void TensorCuda::cleanup_cublas() {
    if (cublas_initialized) {
        cublasDestroy(cublas_handle);
        cublas_initialized = false;
    }
}

void TensorCuda::allocate_gpu_memory() {
    if (size > 0) {
        CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(float)));
        owns_data = true;
    } else {
        d_data = nullptr;
        owns_data = false;
    }
}

void TensorCuda::free_gpu_memory() {
    if (owns_data && d_data != nullptr) {
        CUDA_CHECK(cudaFree(d_data));
    }
    d_data = nullptr;
    owns_data = false;
}

// --- Constructores ---

TensorCuda::TensorCuda() : size(0), d_data(nullptr), owns_data(false) {
    init_cublas();
}

TensorCuda::TensorCuda(const std::vector<int> &shape) : shape(shape), d_data(nullptr), owns_data(false) {
    init_cublas();
    this->size = 1;
    for (int dim : this->shape) {
        if (dim <= 0) {
            this->size = 0;
            break;
        }
        this->size *= dim;
    }
    allocate_gpu_memory();
    zero(); // Inicializar con ceros
}

TensorCuda::TensorCuda(const std::vector<int> &shape, const std::vector<float> &host_data) 
    : shape(shape), d_data(nullptr), owns_data(false) {
    init_cublas();
    this->size = 1;
    for (int dim : this->shape) {
        if (dim <= 0) {
            this->size = 0;
            break;
        }
        this->size *= dim;
    }
    
    if (host_data.size() != static_cast<size_t>(size)) {
        throw std::invalid_argument("El tamaño de los datos no coincide con la forma del tensor");
    }
    
    allocate_gpu_memory();
    copy_from_host(host_data);
}

TensorCuda::TensorCuda(const TensorCuda &other) : shape(other.shape), size(other.size), d_data(nullptr), owns_data(false) {
    allocate_gpu_memory();
    if (size > 0) {
        CUDA_CHECK(cudaMemcpy(d_data, other.d_data, size * sizeof(float), cudaMemcpyDeviceToDevice));
    }
}

TensorCuda& TensorCuda::operator=(const TensorCuda &other) {
    if (this != &other) {
        free_gpu_memory();
        shape = other.shape;
        size = other.size;
        allocate_gpu_memory();
        if (size > 0) {
            CUDA_CHECK(cudaMemcpy(d_data, other.d_data, size * sizeof(float), cudaMemcpyDeviceToDevice));
        }
    }
    return *this;
}

TensorCuda::~TensorCuda() {
    free_gpu_memory();
}

// --- Transferencia de datos ---

void TensorCuda::copy_to_host(std::vector<float> &host_data) const {
    host_data.resize(size);
    if (size > 0) {
        CUDA_CHECK(cudaMemcpy(host_data.data(), d_data, size * sizeof(float), cudaMemcpyDeviceToHost));
    }
}

std::vector<float> TensorCuda::to_host() const {
    std::vector<float> host_data;
    copy_to_host(host_data);
    return host_data;
}

void TensorCuda::copy_from_host(const std::vector<float> &host_data) {
    if (host_data.size() != static_cast<size_t>(size)) {
        throw std::invalid_argument("Tamaño de datos incompatible");
    }
    if (size > 0) {
        CUDA_CHECK(cudaMemcpy(d_data, host_data.data(), size * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void TensorCuda::copy_from_device(const TensorCuda &other) {
    if (size != other.size) {
        throw std::invalid_argument("Tamaños de tensor incompatibles");
    }
    if (size > 0) {
        CUDA_CHECK(cudaMemcpy(d_data, other.d_data, size * sizeof(float), cudaMemcpyDeviceToDevice));
    }
}

// --- Acceso a elementos ---

float TensorCuda::get_element(int r, int c) const {
    if (dims() != 2) {
        throw std::runtime_error("get_element solo funciona para tensores 2D");
    }
    float value;
    int idx = r * shape[1] + c;
    CUDA_CHECK(cudaMemcpy(&value, &d_data[idx], sizeof(float), cudaMemcpyDeviceToHost));
    return value;
}

void TensorCuda::set_element(int r, int c, float value) {
    if (dims() != 2) {
        throw std::runtime_error("set_element solo funciona para tensores 2D");
    }
    int idx = r * shape[1] + c;
    CUDA_CHECK(cudaMemcpy(&d_data[idx], &value, sizeof(float), cudaMemcpyHostToDevice));
}

// --- Operaciones ---

TensorCuda TensorCuda::reshape(const std::vector<int> &new_shape) const {
    int new_size = 1;
    for (int dim : new_shape) {
        new_size *= dim;
    }
    if (new_size != size) {
        throw std::invalid_argument("Reshape: el número total de elementos debe ser el mismo");
    }
    
    TensorCuda result;
    result.shape = new_shape;
    result.size = size;
    result.d_data = d_data; // Compartir datos
    result.owns_data = false; // No posee los datos
    return result;
}

TensorCuda TensorCuda::transpose() const {
    if (dims() != 2) {
        throw std::runtime_error("transpose() sin argumentos solo funciona para tensores 2D");
    }
    
    TensorCuda result({shape[1], shape[0]});
    
    dim3 block(16, 16);
    dim3 grid((shape[1] + block.x - 1) / block.x, (shape[0] + block.y - 1) / block.y);
    
    transpose_2d_kernel<<<grid, block>>>(d_data, result.d_data, shape[0], shape[1]);
    CUDA_CHECK(cudaGetLastError());
    
    return result;
}

TensorCuda TensorCuda::operator*(const TensorCuda &other) const {
    if (dims() != 2 || other.dims() != 2) {
        throw std::runtime_error("La multiplicación de matrices (*) solo está definida para tensores 2D");
    }
    
    int m = shape[0];    // filas de A
    int k = shape[1];    // columnas de A / filas de B
    int n = other.shape[1]; // columnas de B
    
    if (shape[1] != other.shape[0]) {
        throw std::invalid_argument("Multiplicación de matrices: dimensiones incompatibles");
    }
    
    TensorCuda result({m, n});
    
    const float alpha = 1.0f, beta = 0.0f;
    
    // cuBLAS usa column-major, así que necesitamos hacer: C = B^T * A^T = (A * B)^T
    // Luego transponemos el resultado
    cublasStatus_t status = cublasSgemm(cublas_handle,
                                       CUBLAS_OP_N, CUBLAS_OP_N,
                                       n, m, k,
                                       &alpha,
                                       other.d_data, n,
                                       d_data, k,
                                       &beta,
                                       result.d_data, n);
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Error en cuBLAS SGEMM");
    }
    
    return result.transpose();
}

TensorCuda TensorCuda::operator*(float scalar) const {
    TensorCuda result(shape);
    
    int blocks = (size + 255) / 256;
    scalar_multiply_kernel<<<blocks, 256>>>(d_data, scalar, result.d_data, size);
    CUDA_CHECK(cudaGetLastError());
    
    return result;
}

TensorCuda TensorCuda::operator+(const TensorCuda &other) const {
    if (shape != other.shape) {
        throw std::invalid_argument("Suma: las formas de los tensores deben ser idénticas");
    }
    
    TensorCuda result(shape);
    
    int blocks = (size + 255) / 256;
    add_kernel<<<blocks, 256>>>(d_data, other.d_data, result.d_data, size);
    CUDA_CHECK(cudaGetLastError());
    
    return result;
}

TensorCuda TensorCuda::operator-(const TensorCuda &other) const {
    if (shape != other.shape) {
        throw std::invalid_argument("Resta: las formas de los tensores deben ser idénticas");
    }
    
    TensorCuda result(shape);
    
    int blocks = (size + 255) / 256;
    subtract_kernel<<<blocks, 256>>>(d_data, other.d_data, result.d_data, size);
    CUDA_CHECK(cudaGetLastError());
    
    return result;
}

// --- Utilidades ---

void TensorCuda::print() const {
    std::vector<float> host_data = to_host();
    
    std::cout << "Shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i] << (i == shape.size() - 1 ? "" : ", ");
    }
    std::cout << "], Data:" << std::endl;
    
    for (int i = 0; i < size; ++i) {
        std::cout << host_data[i] << " ";
        if (dims() == 2 && (i + 1) % shape[1] == 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}

void TensorCuda::xavier_init(TensorCuda &t) {
    if (t.dims() < 2) return;
    
    float fan_in = static_cast<float>(t.shape[1]);
    float fan_out = static_cast<float>(t.shape[0]);
    float std_dev = std::sqrt(2.0f / (fan_in + fan_out));
    
    // Configurar estados aleatorios
    curandState* states;
    CUDA_CHECK(cudaMalloc(&states, t.size * sizeof(curandState)));
    
    int blocks = (t.size + 255) / 256;
    setup_random_states<<<blocks, 256>>>(states, time(nullptr), t.size);
    CUDA_CHECK(cudaGetLastError());
    
    xavier_init_kernel<<<blocks, 256>>>(t.d_data, states, t.size, std_dev);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaFree(states));
}

void TensorCuda::he_init(TensorCuda &t) {
    if (t.dims() < 2) return;
    
    float fan_in = static_cast<float>(t.shape[1]);
    float std_dev = std::sqrt(2.0f / fan_in);
    
    curandState* states;
    CUDA_CHECK(cudaMalloc(&states, t.size * sizeof(curandState)));
    
    int blocks = (t.size + 255) / 256;
    setup_random_states<<<blocks, 256>>>(states, time(nullptr), t.size);
    CUDA_CHECK(cudaGetLastError());
    
    he_init_kernel<<<blocks, 256>>>(t.d_data, states, t.size, std_dev);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaFree(states));
}

void TensorCuda::zero() {
    int blocks = (size + 255) / 256;
    fill_kernel<<<blocks, 256>>>(d_data, size, 0.0f);
    CUDA_CHECK(cudaGetLastError());
}

float TensorCuda::norm() const {
    thrust::device_ptr<float> dev_ptr(d_data);
    float sum_sq = thrust::transform_reduce(
        dev_ptr, dev_ptr + size,
        [] __device__ (float x) { return x * x; },
        0.0f,
        thrust::plus<float>()
    );
    return std::sqrt(sum_sq);
}

TensorCuda TensorCuda::flatten() const {
    return reshape({1, size});
}

int TensorCuda::argmax() const {
    if (size == 0) return -1;
    
    thrust::device_ptr<float> dev_ptr(d_data);
    auto max_iter = thrust::max_element(dev_ptr, dev_ptr + size);
    return thrust::distance(dev_ptr, max_iter);
}

TensorCuda TensorCuda::slice(int start_row, int end_row, int start_col, int end_col) const {
    if (dims() != 2) {
        throw std::runtime_error("slice solo implementado para tensores 2D");
    }
    
    int new_rows = end_row - start_row;
    int new_cols = end_col - start_col;
    TensorCuda result({new_rows, new_cols});
    
    // Para slice necesitamos copiar datos de manera no contigua
    // Usamos cudaMemcpy2D para esto
    CUDA_CHECK(cudaMemcpy2D(
        result.d_data, new_cols * sizeof(float),
        &d_data[start_row * shape[1] + start_col], shape[1] * sizeof(float),
        new_cols * sizeof(float), new_rows,
        cudaMemcpyDeviceToDevice
    ));
    
    return result;
}

void TensorCuda::set_slice(int start_row, int start_col, const TensorCuda &src) {
    if (dims() != 2 || src.dims() != 2) {
        throw std::runtime_error("set_slice solo implementado para tensores 2D");
    }
    
    auto src_shape = src.get_shape();
    CUDA_CHECK(cudaMemcpy2D(
        &d_data[start_row * shape[1] + start_col], shape[1] * sizeof(float),
        src.d_data, src_shape[1] * sizeof(float),
        src_shape[1] * sizeof(float), src_shape[0],
        cudaMemcpyDeviceToDevice
    ));
}

// --- Batch Matmul ---

TensorCuda batch_matmul(const TensorCuda &a, const TensorCuda &b) {
    auto shape_a = a.get_shape();
    auto shape_b = b.get_shape();
    
    if (shape_a.size() != 3 || shape_b.size() != 3) {
        throw std::runtime_error("Batch matmul solo implementado para tensores 3D");
    }
    if (shape_a[0] != shape_b[0] || shape_a[2] != shape_b[1]) {
        throw std::runtime_error("Dimensiones de batch matmul incompatibles");
    }
    
    int batch_size = shape_a[0];
    int m = shape_a[1];
    int k = shape_a[2];
    int n = shape_b[2];
    
    TensorCuda result({batch_size, m, n});
    
    const float alpha = 1.0f, beta = 0.0f;
    
    // Usar cuBLAS batched GEMM
    cublasStatus_t status = cublasSgemmStridedBatched(
        TensorCuda::cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        b.get_device_ptr(), n, n * k,  // B matrices
        a.get_device_ptr(), k, m * k,  // A matrices  
        &beta,
        result.get_device_ptr(), n, m * n,  // C matrices
        batch_size
    );
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Error en cuBLAS batched SGEMM");
    }
    
    // Transponer cada matriz del resultado
    // Para simplificar, creamos resultado con dimensiones correctas directamente
    TensorCuda final_result({batch_size, m, n});
    
    for (int i = 0; i < batch_size; ++i) {
        TensorCuda a_slice = a.slice(i * m, (i + 1) * m, 0, k);
        TensorCuda b_slice = b.slice(i * k, (i + 1) * k, 0, n);
        TensorCuda c_slice = a_slice * b_slice;
        final_result.set_slice(i * m, 0, c_slice);
    }
    
    return final_result;
}