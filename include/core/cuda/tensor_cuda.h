#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <string>
#include <numeric>
#include <stdexcept>
#include <cassert>
#include <memory>
#include <cuda_runtime.h>
#include <cublas_v2.h>

class Tensor
{
private:
    std::vector<int> shape;
    int size; // Total de elementos
    float* d_data; // Puntero a datos en GPU
    bool owns_data; // Si este tensor posee los datos
    static cublasHandle_t cublas_handle;
    static bool cublas_initialized;

    // Funciones auxiliares privadas
    void allocate_gpu_memory();
    void free_gpu_memory();
    static void init_cublas();
    static void cleanup_cublas();

public:
    // --- Constructores ---
    Tensor();
    Tensor(const std::vector<int> &shape);
    Tensor(const std::vector<int> &shape, const std::vector<float> &host_data);
    
    // Constructor de copia
    Tensor(const Tensor &other);
    
    // Operador de asignación
    Tensor& operator=(const Tensor &other);
    
    // Destructor
    ~Tensor();

    // --- Acceso y Propiedades ---
    const std::vector<int> &get_shape() const { return shape; }
    int get_size() const { return size; }
    int dims() const { return shape.size(); }
    float* get_device_ptr() const { return d_data; }

    // --- Transferencia de datos ---
    void copy_to_host(std::vector<float> &host_data) const;
    std::vector<float> to_host() const;
    void copy_from_host(const std::vector<float> &host_data);
    void copy_from_device(const Tensor &other);

    // Acceso a elementos individuales (lento, solo para debug)
    float get_element(int r, int c) const;
    void set_element(int r, int c, float value);

    // --- Operaciones ---
    Tensor reshape(const std::vector<int> &new_shape) const;
    Tensor transpose() const;
    Tensor transpose(int dim1, int dim2) const;

    // Multiplicación de matrices usando cuBLAS
    Tensor operator*(const Tensor &other) const;
    
    // Multiplicación por escalar
    Tensor operator*(float scalar) const;
    
    // Suma elemento a elemento
    Tensor operator+(const Tensor &other) const;
    
    // Resta elemento a elemento
    Tensor operator-(const Tensor &other) const;

    // --- Utilidades ---
    void print() const;
    static void xavier_init(Tensor &t);
    static void he_init(Tensor &t);
    void zero();
    float norm() const;
    Tensor flatten() const;
    int argmax() const;
    Tensor slice(int start_row, int end_row, int start_col, int end_col) const;
    void set_slice(int start_row, int start_col, const Tensor &src);

    // --- Funciones estáticas de utilidad ---
    static void sync() { cudaDeviceSynchronize(); }
    static void check_cuda_error(cudaError_t error, const char* file, int line);
    static void cleanup() { cleanup_cublas(); }
};

// Función para matmul de lotes
Tensor batch_matmul(const Tensor &a, const Tensor &b);

// Macro para verificar errores de CUDA
#define CUDA_CHECK(call) Tensor::check_cuda_error(call, __FILE__, __LINE__)

#endif // TENSOR_H