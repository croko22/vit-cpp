#include "../../include/model/layernorm.hpp"
#include <cmath>
#include <numeric>

LayerNormalization::LayerNormalization(int feature_size, float epsilon)
    : feature_size_(feature_size),
      epsilon_(epsilon),
      // gamma se inicializa a 1s, beta a 0s.
      gamma_({1, feature_size}),
      beta_({1, feature_size})
{
    // Llenar gamma con 1s y beta con 0s
    std::vector<float> ones(feature_size, 1.0f);
    std::vector<float> zeros(feature_size, 0.0f);
    gamma_.from_vector(ones);
    beta_.from_vector(zeros);
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
    return result;
}