#include "../../include/core/ops.hpp"
#include <stdexcept>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>

Tensor matmul(const Tensor &a, const Tensor &b)
{
    auto shape_a = a.get_shape();
    auto shape_b = b.get_shape();

    if (shape_a.size() != 2 || shape_b.size() != 2 || shape_a[1] != shape_b[0])
    {
        throw std::invalid_argument("Error: Incompatible shapes for matmul.");
    }

    int n = shape_a[0];
    int m = shape_a[1];
    int p = shape_b[1];

    Tensor result({n, p});
    float *data_a = a.get_data();
    float *data_b = b.get_data();
    float *data_res = result.get_data();

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < p; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < m; ++k)
            {
                sum += data_a[i * m + k] * data_b[k * p + j];
            }
            data_res[i * p + j] = sum;
        }
    }
    return result;
}

Tensor softmax(const Tensor &input)
{
    auto shape = input.get_shape();
    if (shape.size() != 2)
    {
        throw std::invalid_argument("Softmax simple solo soporta tensores 2D.");
    }
    int rows = shape[0];
    int cols = shape[1];

    Tensor result(shape);
    float *data_in = input.get_data();
    float *data_out = result.get_data();

    for (int i = 0; i < rows; ++i)
    {
        float *row_start_in = data_in + i * cols;
        float *row_start_out = data_out + i * cols;

        float max_val = *std::max_element(row_start_in, row_start_in + cols);

        float sum_exp = 0.0f;
        for (int j = 0; j < cols; ++j)
        {
            float val = std::exp(row_start_in[j] - max_val);
            row_start_out[j] = val;
            sum_exp += val;
        }

        for (int j = 0; j < cols; ++j)
        {
            row_start_out[j] /= sum_exp;
        }
    }
    return result;
}

Tensor relu(const Tensor &input)
{
    Tensor result(input.get_shape());
    size_t size = input.get_size();
    float *data_in = input.get_data();
    float *data_out = result.get_data();

    for (size_t i = 0; i < size; ++i)
    {
        data_out[i] = std::max(0.0f, data_in[i]);
    }
    return result;
}