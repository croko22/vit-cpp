#include "../../include/core/activation.h"

float Activation::relu(float x)
{
    return std::max(0.0f, x);
}

float Activation::relu_derivative(float x)
{
    return x > 0 ? 1.0f : 0.0f;
}

float Activation::gelu(float x)
{
    return 0.5f * x * (1.0f + tanh(sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
}

float Activation::gelu_derivative(float x)
{
    float tanh_arg = sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x);
    float tanh_val = tanh(tanh_arg);
    float sech_sq = 1.0f - tanh_val * tanh_val;
    return 0.5f * (1.0f + tanh_val) + 0.5f * x * sech_sq * sqrt(2.0f / M_PI) * (1.0f + 0.134145f * x * x);
}

Tensor Activation::apply(const Tensor &input, float (*func)(float))
{
    Tensor result(input.rows, input.cols);
    for (int i = 0; i < input.rows * input.cols; i++)
    {
        result.data[i] = func(input.data[i]);
    }
    return result;
}

Tensor Activation::softmax(const Tensor &input)
{
    Tensor result(input.rows, input.cols);
    for (int i = 0; i < input.rows; i++)
    {
        float max_val = input(i, 0);
        for (int j = 1; j < input.cols; j++)
        {
            max_val = std::max(max_val, input(i, j));
        }
        float sum = 0.0f;
        for (int j = 0; j < input.cols; j++)
        {
            result(i, j) = exp(input(i, j) - max_val);
            sum += result(i, j);
        }
        for (int j = 0; j < input.cols; j++)
        {
            result(i, j) /= sum;
        }
    }
    return result;
}
