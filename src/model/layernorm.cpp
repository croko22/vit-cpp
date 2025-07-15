#include "../../include/model/layernorm.h"

LayerNorm::LayerNorm(int d_mod) : d_model(d_mod), eps(1e-5f),
                                  gamma(1, d_mod), beta(1, d_mod),
                                  gamma_grad(1, d_mod), beta_grad(1, d_mod)
{
    for (int i = 0; i < d_model; i++)
    {
        gamma(0, i) = 1.0f;
        beta(0, i) = 0.0f;
    }
}

Tensor LayerNorm::forward(const Tensor &input)
{
    last_input = input;
    last_mean = Tensor(input.rows, 1);
    last_var = Tensor(input.rows, 1);
    Tensor result(input.rows, input.cols);
    for (int i = 0; i < input.rows; i++)
    {
        float mean = 0.0f;
        for (int j = 0; j < input.cols; j++)
        {
            mean += input(i, j);
        }
        mean /= input.cols;
        last_mean(i, 0) = mean;

        float var = 0.0f;
        for (int j = 0; j < input.cols; j++)
        {
            float diff = input(i, j) - mean;
            var += diff * diff;
        }
        var /= input.cols;
        last_var(i, 0) = var;

        for (int j = 0; j < input.cols; j++)
        {
            float normalized = (input(i, j) - mean) / sqrt(var + eps);
            result(i, j) = gamma(0, j) * normalized + beta(0, j);
        }
    }
    return result;
}

Tensor LayerNorm::backward(const Tensor &grad_output)
{
    return grad_output; // Simplified passthrough
}

void LayerNorm::update(float lr)
{
    zero_grad();
}

void LayerNorm::zero_grad()
{
    gamma_grad.zero();
    beta_grad.zero();
}
