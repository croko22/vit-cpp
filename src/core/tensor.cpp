#include "../../include/core/tensor.h" // Asegúrate que la ruta a tu header sea correcta
#include "../../include/core/random.h" // Y a tu clase de números aleatorios

#include <numeric>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <cmath>

// --- Constructores ---

Tensor::Tensor() : size(0) {}

Tensor::Tensor(const std::vector<int> &shape) : shape(shape)
{
    this->size = 1;
    for (int dim : this->shape)
    {
        if (dim <= 0)
        { // Las dimensiones no pueden ser cero o negativas
            this->size = 0;
            break;
        }
        this->size *= dim;
    }
    this->data.resize(this->size, 0.0f);
}

// En tensor.cpp

// ... resto de tu código ...

// Implementación de los getters
std::vector<float> &Tensor::get_data()
{
    return this->data;
}

const std::vector<float> &Tensor::get_data() const
{
    return this->data;
}

float &Tensor::operator()(int r, int c)
{
    if (this->dims() != 2)
        throw std::runtime_error("El operador (r,c) solo funciona para tensores 2D.");
    return this->data[r * this->shape[1] + c];
}

const float &Tensor::operator()(int r, int c) const
{
    if (this->dims() != 2)
        throw std::runtime_error("El operador (r,c) solo funciona para tensores 2D.");
    return this->data[r * this->shape[1] + c];
}

Tensor Tensor::reshape(const std::vector<int> &new_shape) const
{
    int new_size = 1;
    for (int dim : new_shape)
        new_size *= dim;
    if (new_size != this->size)
        throw std::invalid_argument("Reshape: el número total de elementos debe ser el mismo.");
    Tensor result = *this;
    result.shape = new_shape;
    return result;
}

Tensor Tensor::transpose() const
{
    if (this->dims() != 2)
    {
        throw std::runtime_error("transpose() sin argumentos solo funciona para tensores 2D.");
    }
    return this->transpose(0, 1);
}

Tensor Tensor::transpose(int dim1, int dim2) const
{
    if (dim1 >= dims() || dim2 >= dims() || dim1 < 0 || dim2 < 0)
    {
        throw std::invalid_argument("Dimensiones de transpose inválidas.");
    }

    std::vector<int> new_shape = this->shape;
    std::swap(new_shape[dim1], new_shape[dim2]);
    Tensor result(new_shape);

    // Calcula los strides para la forma original
    std::vector<int> original_strides(dims());
    original_strides.back() = 1;
    for (int i = dims() - 2; i >= 0; --i)
    {
        original_strides[i] = original_strides[i + 1] * shape[i + 1];
    }

    // Calcula los strides para la NUEVA forma
    std::vector<int> new_strides(dims());
    new_strides.back() = 1;
    for (int i = dims() - 2; i >= 0; --i)
    {
        new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
    }

    std::vector<int> current_pos(dims(), 0);
    for (int i = 0; i < size; ++i)
    {
        // Calcula el índice original basado en la posición actual y los strides originales
        int original_index = 0;
        for (int d = 0; d < dims(); ++d)
            original_index += current_pos[d] * original_strides[d];

        // Transpone la posición actual para encontrar la nueva posición
        std::vector<int> transp_pos = current_pos;
        std::swap(transp_pos[dim1], transp_pos[dim2]);

        // Calcula el nuevo índice basado en la posición transpuesta y los nuevos strides
        int new_index = 0;
        for (int d = 0; d < dims(); ++d)
            new_index += transp_pos[d] * new_strides[d];

        result.data[new_index] = this->data[original_index];

        // Avanza a la siguiente posición en el tensor original
        for (int d = dims() - 1; d >= 0; --d)
        {
            if (++current_pos[d] < shape[d])
                break;
            current_pos[d] = 0;
        }
    }
    return result;
}

Tensor Tensor::operator+(const Tensor &other) const
{
    if (this->shape != other.shape)
    {
        throw std::invalid_argument("Suma: las formas de los tensores deben ser idénticas.");
    }
    Tensor result(this->shape);
    for (int i = 0; i < this->size; i++)
    {
        result.data[i] = this->data[i] + other.data[i];
    }
    return result;
}

// Multiplicación de matrices (solo para 2D)
Tensor Tensor::operator*(const Tensor &other) const
{
    if (this->dims() != 2 || other.dims() != 2)
    {
        throw std::runtime_error("La multiplicación de matrices (*) solo está definida para tensores 2D.");
    }
    int r1 = this->shape[0], c1 = this->shape[1];
    int r2 = other.shape[0], c2 = other.shape[1];
    if (c1 != r2)
    {
        throw std::invalid_argument("Multiplicación de matrices: dimensiones incompatibles.");
    }

    Tensor result({r1, c2});
    for (int i = 0; i < r1; ++i)
    {
        for (int j = 0; j < c2; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < c1; ++k)
            {
                sum += (*this)(i, k) * other(k, j);
            }
            result(i, j) = sum;
        }
    }
    return result;
}

Tensor Tensor::operator*(float scalar) const
{
    Tensor result(this->shape);
    for (int i = 0; i < this->size; i++)
    {
        result.data[i] = this->data[i] * scalar;
    }
    return result;
}

// --- Batch Matmul (Función externa para claridad) ---

Tensor batch_matmul(const Tensor &a, const Tensor &b)
{
    auto shape_a = a.get_shape();
    auto shape_b = b.get_shape();

    if (shape_a.size() != 3 || shape_b.size() != 3)
    {
        throw std::runtime_error("Batch matmul solo implementado para tensores 3D.");
    }
    if (shape_a[0] != shape_b[0] || shape_a[2] != shape_b[1])
    {
        throw std::runtime_error("Dimensiones de batch matmul incompatibles.");
    }

    int batch_size = shape_a[0];
    int rows_a = shape_a[1];
    int cols_a = shape_a[2]; // inner_dim
    int cols_b = shape_b[2];

    Tensor result({batch_size, rows_a, cols_b});

    int slice_a_size = rows_a * cols_a;
    int slice_b_size = cols_a * cols_b;
    int slice_res_size = rows_a * cols_b;

    // Itera sobre cada matriz en el batch
    for (int i = 0; i < batch_size; ++i)
    {
        // Realiza la multiplicación de matrices estándar para el slice actual
        for (int j = 0; j < rows_a; ++j)
        {
            for (int k = 0; k < cols_b; ++k)
            {
                float sum = 0.0f;
                for (int l = 0; l < cols_a; ++l)
                {
                    sum += a.data[i * slice_a_size + j * cols_a + l] * b.data[i * slice_b_size + l * cols_b + k];
                }
                result.data[i * slice_res_size + j * cols_b + k] = sum;
            }
        }
    }
    return result;
}

// --- Utilidades (ejemplos) ---

void Tensor::print() const
{
    // Implementación de print más robusta para N-D sería recursiva,
    // pero para depurar una versión simple puede ser suficiente.
    std::cout << "Shape: [";
    for (size_t i = 0; i < shape.size(); ++i)
    {
        std::cout << shape[i] << (i == shape.size() - 1 ? "" : ", ");
    }
    std::cout << "], Data:" << std::endl;
    for (int i = 0; i < size; ++i)
    {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
}

void Tensor::xavier_init(Tensor &t)
{
    if (t.dims() < 2)
        return; // No tiene sentido para 1D o 0D
    float fan_in = t.get_shape()[1];
    float fan_out = t.get_shape()[0];
    float std = sqrt(2.0f / (fan_in + fan_out));
    for (float &val : t.data)
    {
        val = Random::randn(0.0f, std);
    }
}

// En tensor.cpp

void Tensor::zero()
{
    std::fill(data.begin(), data.end(), 0.0f);
}

float Tensor::norm() const
{
    float sum_sq = 0.0f;
    for (float val : data)
    {
        sum_sq += val * val;
    }
    return std::sqrt(sum_sq);
}

// en tensor.cpp

Tensor Tensor::flatten() const
{
    // Aplana el tensor a una forma 2D de [1, N]
    return this->reshape({1, this->size});
}

int Tensor::argmax() const
{
    if (data.empty())
    {
        return -1; // O lanzar un error
    }
    // Encuentra el índice del valor máximo en el vector de datos plano
    auto max_it = std::max_element(data.begin(), data.end());
    return std::distance(data.begin(), max_it);
}

// en tensor.cpp

Tensor Tensor::slice(int start_row, int end_row, int start_col, int end_col) const
{
    if (dims() != 2)
        throw std::runtime_error("slice solo implementado para tensores 2D.");

    int new_rows = end_row - start_row;
    int new_cols = end_col - start_col;
    Tensor result({new_rows, new_cols});

    for (int i = 0; i < new_rows; ++i)
    {
        for (int j = 0; j < new_cols; ++j)
        {
            result(i, j) = (*this)(start_row + i, start_col + j);
        }
    }
    return result;
}

void Tensor::set_slice(int start_row, int start_col, const Tensor &src)
{
    if (dims() != 2 || src.dims() != 2)
        throw std::runtime_error("set_slice solo implementado para tensores 2D.");

    auto src_shape = src.get_shape();
    for (int i = 0; i < src_shape[0]; ++i)
    {
        for (int j = 0; j < src_shape[1]; ++j)
        {
            (*this)(start_row + i, start_col + j) = src(i, j);
        }
    }
}