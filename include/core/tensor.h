#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <cassert>

// Forward declaration of Random for xavier_init and he_init
class Random;

class Tensor
{
public:
    std::vector<float> data;
    int rows, cols;
    Tensor();
    Tensor(int r, int c);
    Tensor(const std::vector<std::vector<float>> &d);
    float &operator()(int i, int j);
    const float &operator()(int i, int j) const;
    Tensor operator+(const Tensor &other) const;
    Tensor operator-(const Tensor &other) const;
    Tensor operator*(const Tensor &other) const;
    Tensor operator*(float scalar) const;
    Tensor transpose() const;
    void zero();
    void xavier_init();
    void he_init();
    static Tensor eye(int n);
    void print() const;
    Tensor slice(int start_row, int end_row, int start_col, int end_col) const;
    void set_slice(int start_row, int start_col, const Tensor &src);
    Tensor hadamard(const Tensor &other) const;
    Tensor row_normalize() const;
};

#endif // TENSOR_H