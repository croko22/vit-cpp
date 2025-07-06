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

Tensor softmax_backward(const Tensor &grad_output, const Tensor &softmax_output)
{
    auto shape = softmax_output.get_shape();
    if (shape.size() != 2 || shape != grad_output.get_shape())
    {
        throw std::invalid_argument("Softmax backward requires matching shapes.");
    }

    int rows = shape[0];
    int cols = shape[1];

    Tensor result(shape);
    float *data_grad = grad_output.get_data();
    float *data_softmax = softmax_output.get_data();
    float *data_result = result.get_data();

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            data_result[i * cols + j] = data_grad[i * cols + j] * data_softmax[i * cols + j] *
                                        (1.0f - data_softmax[i * cols + j]);
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

std::pair<Tensor, Tensor> matmul_backward(const Tensor &grad_output, const Tensor &a, const Tensor &b)
{
    // grad_a = grad_output @ b.T
    Tensor grad_a = matmul(grad_output, b.transpose());
    // grad_b = a.T @ grad_output
    Tensor grad_b = matmul(a.transpose(), grad_output);
    return {grad_a, grad_b};
}

Tensor relu_backward(const Tensor &grad_output, const Tensor &input)
{
    Tensor grad_input(input.get_shape());
    float *grad_out_data = grad_output.get_data();
    float *input_data = input.get_data();
    float *grad_in_data = grad_input.get_data();

    for (size_t i = 0; i < input.get_size(); ++i)
    {
        // El gradiente solo fluye si la entrada original era > 0
        grad_in_data[i] = (input_data[i] > 0) ? grad_out_data[i] : 0.0f;
    }
    return grad_input;
}

Tensor sum(const Tensor &input, int axis, bool keep_dims)
{
    const auto &in_shape = input.get_shape();
    if (in_shape.size() != 2 || axis != 0)
    {
        throw std::runtime_error("La función de suma actual solo soporta axis=0 para tensores 2D.");
    }

    // Aseguramos que la forma de salida sea [1, N], que es 2D.
    std::vector<int> out_shape = {1, in_shape[1]};
    Tensor output(out_shape);
    output.zero_data();

    const float *in_data = input.get_data();
    float *out_data = output.get_data();
    int rows = in_shape[0];
    int cols = in_shape[1];
    for (int j = 0; j < cols; ++j)
    {
        float col_sum = 0.0f;
        for (int i = 0; i < rows; ++i)
        {
            col_sum += in_data[i * cols + j];
        }
        out_data[j] = col_sum;
    }
    return output;
}