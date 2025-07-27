#include "../../include/model/layernorm.h"
#include "../../include/core/optimizer.h"
#include <cmath>
#include <numeric>

LayerNorm::LayerNorm(int d_mod)
    : d_model(d_mod),
      eps(1e-5f),

      gamma({d_mod}),
      beta({d_mod}),
      gamma_grad({d_mod}),
      beta_grad({d_mod})
{

    auto &gamma_data = gamma.get_data();
    auto &beta_data = beta.get_data();
    std::fill(gamma_data.begin(), gamma_data.end(), 1.0f);
    std::fill(beta_data.begin(), beta_data.end(), 0.0f);
}

Tensor LayerNorm::forward(const Tensor &input)
{
    last_input = input;
    auto shape = input.get_shape();
    Tensor result(shape);

    int last_dim = shape.back();
    if (last_dim != d_model)
    {
        throw std::runtime_error("La última dimensión del input no coincide con d_model de LayerNorm.");
    }

    int outer_size = input.get_size() / last_dim;
    last_mean = Tensor({outer_size});
    last_var = Tensor({outer_size});

    auto &result_data = result.get_data();
    const auto &input_data = input.get_data();
    auto &mean_data = last_mean.get_data();
    auto &var_data = last_var.get_data();
    const auto &gamma_data = gamma.get_data();
    const auto &beta_data = beta.get_data();

    for (int i = 0; i < outer_size; ++i)
    {
        int offset = i * last_dim;

        float sum = 0.0f;
        for (int j = 0; j < last_dim; ++j)
        {
            sum += input_data[offset + j];
        }
        float mean = sum / last_dim;
        mean_data[i] = mean;

        float var_sum = 0.0f;
        for (int j = 0; j < last_dim; ++j)
        {
            float diff = input_data[offset + j] - mean;
            var_sum += diff * diff;
        }
        float var = var_sum / last_dim;
        var_data[i] = var;

        float inv_std = 1.0f / std::sqrt(var + eps);

        for (int j = 0; j < last_dim; ++j)
        {
            float normalized = (input_data[offset + j] - mean) * inv_std;
            result_data[offset + j] = gamma_data[j] * normalized + beta_data[j];
        }
    }
    return result;
}

Tensor LayerNorm::backward(const Tensor &grad_output)
{
    auto shape = last_input.get_shape();
    Tensor grad_input(shape);

    int last_dim = d_model;
    int outer_size = last_input.get_size() / last_dim;

    const auto &x_data = last_input.get_data();
    const auto &grad_out_data = grad_output.get_data();
    auto &grad_in_data = grad_input.get_data();

    const auto &mean_data = last_mean.get_data();
    const auto &var_data = last_var.get_data();
    const auto &gamma_data = gamma.get_data();
    auto &g_grad_data = gamma_grad.get_data();
    auto &b_grad_data = beta_grad.get_data();

    for (int i = 0; i < outer_size; ++i)
    {
        int offset = i * last_dim;
        float mean = mean_data[i];
        float var = var_data[i];
        float inv_std = 1.0f / std::sqrt(var + eps);

        for (int j = 0; j < last_dim; ++j)
        {
            float x_hat = (x_data[offset + j] - mean) * inv_std;
            g_grad_data[j] += grad_out_data[offset + j] * x_hat;
            b_grad_data[j] += grad_out_data[offset + j];
        }

        float sum1 = 0.0f;
        float sum2 = 0.0f;
        for (int j = 0; j < last_dim; ++j)
        {
            float grad_x_hat_j = grad_out_data[offset + j] * gamma_data[j];
            sum1 += grad_x_hat_j;
            sum2 += grad_x_hat_j * (x_data[offset + j] - mean);
        }

        for (int j = 0; j < last_dim; ++j)
        {
            float grad_x_hat_j = grad_out_data[offset + j] * gamma_data[j];
            grad_in_data[offset + j] = (1.0f / last_dim) * inv_std * (last_dim * grad_x_hat_j - sum1 - (x_data[offset + j] - mean) * inv_std * inv_std * sum2);
        }
    }

    return grad_input;
}

std::vector<Parameter> LayerNorm::get_parameters()
{
    return {
        {&gamma, &gamma_grad},
        {&beta, &beta_grad}};
}

void LayerNorm::zero_grad()
{
    gamma_grad.zero();
    beta_grad.zero();
}