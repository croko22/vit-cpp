#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <string>
#include <numeric>
#include <stdexcept>
#include <cassert>

class Tensor
{
private:
    std::vector<int> shape;
    int size; // Total de elementos, para evitar recalcularlo

public:
    std::vector<float> data;
    std::vector<float> &get_data();
    const std::vector<float> &get_data() const;
    // --- Constructores ---
    Tensor();
    Tensor(const std::vector<int> &shape);
    Tensor(const std::vector<int> &shape, const std::vector<float> &data);

    // --- Acceso y Propiedades ---
    const std::vector<int> &get_shape() const { return shape; }
    int get_size() const { return size; }
    int dims() const { return shape.size(); }

    // Acceso a los datos (simplificado para 2D, lanza error para otros)
    float &operator()(int r, int c);
    const float &operator()(int r, int c) const;

    // --- Operaciones ---
    Tensor reshape(const std::vector<int> &new_shape) const;
    Tensor transpose() const;
    Tensor transpose(int dim1, int dim2) const;

    // Multiplicación de matrices estándar (para capas lineales)
    Tensor operator*(const Tensor &other) const;

    // Multiplicación por escalar
    Tensor operator*(float scalar) const;

    // Suma elemento a elemento
    Tensor operator+(const Tensor &other) const;

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
};

// Declaración de una función para matmul de lotes, fuera de la clase
Tensor batch_matmul(const Tensor &a, const Tensor &b);

#endif // TENSOR_H