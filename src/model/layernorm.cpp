#include "../../include/model/layernorm.h" // Ajusta tu ruta
#include <cmath>
#include <numeric>

// --- Constructor Corregido ---
LayerNorm::LayerNorm(int d_mod)
    : d_model(d_mod),
      eps(1e-5f),
      // CORREGIDO: gamma y beta son vectores 1D
      gamma({d_mod}),
      beta({d_mod}),
      gamma_grad({d_mod}),
      beta_grad({d_mod})
{
    // CORREGIDO: Inicialización de los vectores 1D
    auto &gamma_data = gamma.get_data();
    auto &beta_data = beta.get_data();
    std::fill(gamma_data.begin(), gamma_data.end(), 1.0f);
    std::fill(beta_data.begin(), beta_data.end(), 0.0f);
}

// --- Forward Pass N-Dimensional ---
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

    // Itera sobre cada "fila" o "slice" del tensor
    for (int i = 0; i < outer_size; ++i)
    {
        int offset = i * last_dim;

        // 1. Calcula la media
        float sum = 0.0f;
        for (int j = 0; j < last_dim; ++j)
        {
            sum += input_data[offset + j];
        }
        float mean = sum / last_dim;
        mean_data[i] = mean;

        // 2. Calcula la varianza
        float var_sum = 0.0f;
        for (int j = 0; j < last_dim; ++j)
        {
            float diff = input_data[offset + j] - mean;
            var_sum += diff * diff;
        }
        float var = var_sum / last_dim;
        var_data[i] = var;

        float inv_std = 1.0f / std::sqrt(var + eps);

        // 3. Normaliza y aplica gamma y beta
        for (int j = 0; j < last_dim; ++j)
        {
            float normalized = (input_data[offset + j] - mean) * inv_std;
            result_data[offset + j] = gamma_data[j] * normalized + beta_data[j];
        }
    }
    return result;
}

// --- Backward Pass N-Dimensional ---
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

        // 1. Acumula gradientes para gamma y beta (sobre todo el batch/secuencia)
        for (int j = 0; j < last_dim; ++j)
        {
            float x_hat = (x_data[offset + j] - mean) * inv_std;
            g_grad_data[j] += grad_out_data[offset + j] * x_hat;
            b_grad_data[j] += grad_out_data[offset + j];
        }

        // 2. Calcula gradiente para la entrada x
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

// en src/model/layernorm.cpp
void LayerNorm::update(float lr, int batch_size)
{
    auto &gamma_data = gamma.get_data();
    auto &beta_data = beta.get_data();
    const auto &g_grad_data = gamma_grad.get_data();
    const auto &b_grad_data = beta_grad.get_data();

    // Escala el learning rate por el tamaño del lote para promediar el gradiente
    float scale = lr / batch_size;

    for (int j = 0; j < d_model; ++j)
    {
        gamma_data[j] -= scale * g_grad_data[j];
        beta_data[j] -= scale * b_grad_data[j];
    }
}
void LayerNorm::zero_grad()
{
    gamma_grad.zero();
    beta_grad.zero();
}