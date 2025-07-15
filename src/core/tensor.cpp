#include "../../include/core/tensor.h"
#include "../../include/core/random.h"

Tensor::Tensor() : rows(0), cols(0) {}

Tensor::Tensor(int r, int c) : rows(r), cols(c)
{
    data.resize(rows * cols, 0.0f);
}

Tensor::Tensor(const std::vector<std::vector<float>> &d) : rows(d.size()), cols(d[0].size())
{
    data.resize(rows * cols);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            data[i * cols + j] = d[i][j];
        }
    }
}

float &Tensor::operator()(int i, int j)
{
    return data[i * cols + j];
}

const float &Tensor::operator()(int i, int j) const
{
    return data[i * cols + j];
}

Tensor Tensor::operator+(const Tensor &other) const
{
    assert(rows == other.rows && cols == other.cols);
    Tensor result(rows, cols);
    for (int i = 0; i < rows * cols; i++)
    {
        result.data[i] = data[i] + other.data[i];
    }
    return result;
}

Tensor Tensor::operator-(const Tensor &other) const
{
    assert(rows == other.rows && cols == other.cols);
    Tensor result(rows, cols);
    for (int i = 0; i < rows * cols; i++)
    {
        result.data[i] = data[i] - other.data[i];
    }
    return result;
}

Tensor Tensor::operator*(const Tensor &other) const
{
    assert(cols == other.rows);
    Tensor result(rows, other.cols);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < other.cols; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < cols; k++)
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
    Tensor result(rows, cols);
    for (int i = 0; i < rows * cols; i++)
    {
        result.data[i] = data[i] * scalar;
    }
    return result;
}

Tensor Tensor::transpose() const
{
    Tensor result(cols, rows);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            result(j, i) = (*this)(i, j);
        }
    }
    return result;
}

void Tensor::zero()
{
    std::fill(data.begin(), data.end(), 0.0f);
}

void Tensor::xavier_init()
{
    float std = sqrt(2.0f / (rows + cols));
    for (float &val : data)
    {
        val = Random::randn(0.0f, std);
    }
}

void Tensor::he_init()
{
    float std = sqrt(2.0f / rows);
    for (float &val : data)
    {
        val = Random::randn(0.0f, std);
    }
}

Tensor Tensor::eye(int n)
{
    Tensor result(n, n);
    for (int i = 0; i < n; i++)
    {
        result(i, i) = 1.0f;
    }
    return result;
}

void Tensor::print() const
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            std::cout << (*this)(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

Tensor Tensor::slice(int start_row, int end_row, int start_col, int end_col) const
{
    Tensor result(end_row - start_row, end_col - start_col);
    for (int i = start_row; i < end_row; i++)
    {
        for (int j = start_col; j < end_col; j++)
        {
            result(i - start_row, j - start_col) = (*this)(i, j);
        }
    }
    return result;
}

void Tensor::set_slice(int start_row, int start_col, const Tensor &src)
{
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            (*this)(start_row + i, start_col + j) = src(i, j);
        }
    }
}

Tensor Tensor::hadamard(const Tensor &other) const
{
    assert(rows == other.rows && cols == other.cols);
    Tensor result(rows, cols);
    for (int i = 0; i < rows * cols; i++)
    {
        result.data[i] = data[i] * other.data[i];
    }
    return result;
}

Tensor Tensor::row_normalize() const
{
    Tensor result(rows, cols);
    for (int i = 0; i < rows; i++)
    {
        float sum = 0.0f;
        for (int j = 0; j < cols; j++)
        {
            sum += (*this)(i, j) * (*this)(i, j);
        }
        float norm = sqrt(sum + 1e-8f);
        for (int j = 0; j < cols; j++)
        {
            result(i, j) = (*this)(i, j) / norm;
        }
    }
    return result;
}
