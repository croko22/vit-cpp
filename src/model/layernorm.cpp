#include "../../include/model/layernorm.hpp"
#include "../../include/core/ops.hpp"
#include <cmath>
#include <numeric>

LayerNormalization::LayerNormalization(int feature_size, float epsilon)
    : gamma_({1, feature_size}),
      beta_({1, feature_size}),
      epsilon_(epsilon),
      feature_size_(feature_size)
{
    // Llenar gamma con 1s y beta con 0s
    std::vector<float> ones(feature_size, 1.0f);
    std::vector<float> zeros(feature_size, 0.0f);
    gamma_.from_vector(ones);
    beta_.from_vector(zeros);
    gamma_.init_grad(); // 👈 AÑADE ESTO
    beta_.init_grad();  // 👈 Y ESTO TAMBIÉN
}

// Implementación simple de LayerNorm. Asume input [N, D]
Tensor LayerNormalization::forward(const Tensor &input)
{
    auto shape = input.get_shape();
    int rows = shape[0];
    int cols = shape[1]; // feature_size

    Tensor result(shape);
    float *data_in = input.get_data();
    float *data_out = result.get_data();
    float *gamma_data = gamma_.get_data();
    float *beta_data = beta_.get_data();

    for (int i = 0; i < rows; ++i)
    {
        float *row_start = data_in + i * cols;

        // 1. Calcular media
        float mean = std::accumulate(row_start, row_start + cols, 0.0f) / cols;

        // 2. Calcular varianza
        float variance = 0.0f;
        for (int j = 0; j < cols; ++j)
        {
            variance += std::pow(row_start[j] - mean, 2);
        }
        variance /= cols;

        // 3. Normalizar
        float inv_std = 1.0f / std::sqrt(variance + epsilon_);
        for (int j = 0; j < cols; ++j)
        {
            float normalized = (row_start[j] - mean) * inv_std;
            // 4. Escalar y desplazar
            data_out[i * cols + j] = normalized * gamma_data[j] + beta_data[j];
        }
    }

    // Guardar en caché para backward
    input_cache_ = input;
    normalized_input_cache_ = result;

    Tensor mean_tensor({rows});
    Tensor var_tensor({rows});
    float *mean_data = mean_tensor.get_data();
    float *var_data = var_tensor.get_data();

    for (int i = 0; i < rows; ++i)
    {
        float *row_start = data_in + i * cols;
        mean_data[i] = std::accumulate(row_start, row_start + cols, 0.0f) / cols;

        float variance = 0.0f;
        for (int j = 0; j < cols; ++j)
            variance += std::pow(row_start[j] - mean_data[i], 2);
        var_data[i] = variance / cols;
    }

    mean_cache_ = mean_tensor;
    var_cache_ = var_tensor;

    return result;
}

Tensor LayerNormalization::backward(const Tensor &grad_output)
{
    int rows = grad_output.get_shape()[0];
    int cols = grad_output.get_shape()[1];

    // 1. Gradientes de los parámetros gamma y beta
    // dL/dbeta = dL/dy
    *(beta_.grad_) = grad_output; // Simplificado, debería sumar gradientes por batch
    // dL/dgamma = dL/dy * x_norm
    *(gamma_.grad_) = matmul(grad_output, normalized_input_cache_.transpose());

    // 2. Propagar gradiente a la entrada normalizada (x_norm)
    // dL/dx_norm = dL/dy * gamma
    Tensor grad_norm = matmul(grad_output, gamma_.transpose());

    // 3. Propagar gradiente a través de la normalización (la parte más compleja)
    Tensor grad_input(input_cache_.get_shape());
    float *grad_norm_data = grad_norm.get_data();
    // float *norm_data = normalized_input_cache_.get_data();
    float *var_data = var_cache_.get_data();
    float *grad_in_data = grad_input.get_data();

    for (int i = 0; i < rows; ++i)
    {
        float inv_std = 1.0f / std::sqrt(var_data[i] + epsilon_);

        float dL_dvar = 0.0;
        for (int j = 0; j < cols; ++j)
        {
            dL_dvar += grad_norm_data[i * cols + j] * (input_cache_.get_data()[i * cols + j] - mean_cache_.get_data()[i]) * (-0.5f * std::pow(inv_std, 3));
        }

        float dL_dmean = 0.0;
        for (int j = 0; j < cols; ++j)
        {
            dL_dmean += grad_norm_data[i * cols + j] * (-inv_std);
        }

        for (int j = 0; j < cols; ++j)
        {
            float d_norm_dx = inv_std;
            float d_var_dx = (2.0f * (input_cache_.get_data()[i * cols + j] - mean_cache_.get_data()[i])) / cols;
            float d_mean_dx = 1.0f / cols;

            grad_in_data[i * cols + j] = grad_norm_data[i * cols + j] * d_norm_dx + dL_dvar * d_var_dx + dL_dmean * d_mean_dx;
        }
    }

    return grad_input;
}

//? WORKING ONE BUT NAN
// Tensor LayerNormalization::backward(const Tensor &grad_output)
// {
//     int rows = grad_output.get_shape()[0];
//     int cols = grad_output.get_shape()[1];

//     // --- Gradientes gamma y beta manuales ---
//     float *grad_gamma_data = gamma_.grad_->get_data();
//     float *grad_beta_data = beta_.grad_->get_data();
//     float *grad_out_data = grad_output.get_data();
//     float *x_norm_data = normalized_input_cache_.get_data();

//     std::fill(grad_gamma_data, grad_gamma_data + cols, 0.0f);
//     std::fill(grad_beta_data, grad_beta_data + cols, 0.0f);

//     for (int i = 0; i < rows; ++i)
//     {
//         for (int j = 0; j < cols; ++j)
//         {
//             grad_gamma_data[j] += grad_out_data[i * cols + j] * x_norm_data[i * cols + j];
//             grad_beta_data[j]  += grad_out_data[i * cols + j];
//         }
//     }

//     // --- dL/dx_norm = dL/dy * gamma ---
//     Tensor grad_norm(grad_output.get_shape());
//     float *grad_norm_data = grad_norm.get_data();
//     float *gamma_data = gamma_.get_data();
//     for (int i = 0; i < rows; ++i)
//         for (int j = 0; j < cols; ++j)
//             grad_norm_data[i * cols + j] = grad_out_data[i * cols + j] * gamma_data[j];

//     // --- Retropropagación completa ---
//     Tensor grad_input(grad_output.get_shape());
//     float *grad_in_data = grad_input.get_data();
//     float *x_data = input_cache_.get_data();
//     float *mean_data = mean_cache_.get_data();
//     float *var_data = var_cache_.get_data();

//     for (int i = 0; i < rows; ++i)
//     {
//         float inv_std = 1.0f / std::sqrt(var_data[i] + epsilon_);

//         float dL_dvar = 0.0f;
//         float dL_dmean = 0.0f;

//         for (int j = 0; j < cols; ++j)
//         {
//             float x_ij = x_data[i * cols + j];
//             float mu_i = mean_data[i];
//             float gnorm = grad_norm_data[i * cols + j];
//             dL_dvar += gnorm * (x_ij - mu_i) * (-0.5f * std::pow(inv_std, 3));
//             dL_dmean += gnorm * (-inv_std);
//         }

//         for (int j = 0; j < cols; ++j)
//         {
//             float x_ij = x_data[i * cols + j];
//             float mu_i = mean_data[i];
//             float gnorm = grad_norm_data[i * cols + j];

//             float d_norm_dx = inv_std;
//             float d_var_dx = 2.0f * (x_ij - mu_i) / cols;
//             float d_mean_dx = 1.0f / cols;

//             grad_in_data[i * cols + j] = gnorm * d_norm_dx + dL_dvar * d_var_dx + dL_dmean * d_mean_dx;
//         }
//     }

//     return grad_input;
// }

void LayerNormalization::zero_all_grads()
{
    gamma_.zero_grad();
    beta_.zero_grad();
}

void LayerNormalization::get_parameters(std::vector<Tensor *> &params)
{
    params.push_back(&gamma_);
    params.push_back(&beta_);
}