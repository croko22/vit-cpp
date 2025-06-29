#include "../../include/model/feedforward.hpp"
#include "../../include/core/ops.hpp"
#include <iostream>
#include <random>
#include <cmath>

FeedForwardNetwork::FeedForwardNetwork(int d_model, int d_ff)
    : w1_({d_model, d_ff}),
      b1_({1, d_ff}), // Bias es un vector fila
      w2_({d_ff, d_model}),
      b2_({1, d_model})
{
    std::mt19937 gen(std::random_device{}());

    // He initialization para ReLU (mejor que Xavier para activaciones ReLU)
    float he_std_w1 = std::sqrt(2.0f / d_model);
    std::normal_distribution<float> dist_w1(0.0f, he_std_w1);

    // Xavier initialization para la capa de salida (sin activación)
    float xavier_std_w2 = std::sqrt(1.0f / d_ff);
    std::normal_distribution<float> dist_w2(0.0f, xavier_std_w2);

    // Inicializar W1 (entrada -> hidden con ReLU)
    std::vector<float> w1_data(d_model * d_ff);
    for (auto &val : w1_data)
    {
        val = dist_w1(gen);
    }
    w1_.from_vector(w1_data);

    // Inicializar W2 (hidden -> salida, sin activación)
    std::vector<float> w2_data(d_ff * d_model);
    for (auto &val : w2_data)
    {
        val = dist_w2(gen);
    }
    w2_.from_vector(w2_data);

    // Bias inicializados a cero (estándar en Transformers)
    std::vector<float> b1_data(d_ff, 0.0f);
    std::vector<float> b2_data(d_model, 0.0f);
    b1_.from_vector(b1_data);
    b2_.from_vector(b2_data);

    std::cout << "Transformer FFN initialized: d_model=" << d_model
              << ", d_ff=" << d_ff << " (expansion factor: "
              << (float)d_ff / d_model << "x)" << std::endl;
}

Tensor FeedForwardNetwork::forward(const Tensor &input)
{
    std::cout << "=== DEBUG FeedForward Forward ===" << std::endl;

    // Debug: imprimir formas
    const auto &input_shape = input.get_shape();
    const auto &w1_shape = w1_.get_shape();
    const auto &b1_shape = b1_.get_shape();

    std::cout << "Input shape: [" << input_shape[0] << ", " << input_shape[1] << "]" << std::endl;
    std::cout << "W1 shape: [" << w1_shape[0] << ", " << w1_shape[1] << "]" << std::endl;
    std::cout << "B1 shape: [" << b1_shape[0] << ", " << b1_shape[1] << "]" << std::endl;

    // 1. Primera proyección lineal
    Tensor hidden = matmul(input, w1_);

    std::cout << "Hidden shape after matmul: [" << hidden.get_shape()[0] << ", " << hidden.get_shape()[1] << "]" << std::endl;

    // SOLUCIÓN: Suma fila por fila si el bias es [1, d_ff]
    // O implementa broadcasting en tu clase Tensor

    // Opción 1: Suma manual (temporal)
    std::vector<float> hidden_data = hidden.to_vector();
    std::vector<float> bias_data = b1_.to_vector();

    int seq_len = hidden.get_shape()[0];
    int d_ff = hidden.get_shape()[1];

    for (int i = 0; i < seq_len; ++i)
    {
        for (int j = 0; j < d_ff; ++j)
        {
            hidden_data[i * d_ff + j] += bias_data[j];
        }
    }

    hidden.from_vector(hidden_data);

    // 2. Aplicar ReLU
    Tensor activated = relu(hidden);

    // 3. Segunda proyección lineal + bias
    Tensor output = matmul(activated, w2_);

    // Mismo problema con b2_, aplicar la misma solución
    std::vector<float> output_data = output.to_vector();
    std::vector<float> bias2_data = b2_.to_vector();

    int d_model = output.get_shape()[1];

    for (int i = 0; i < seq_len; ++i)
    {
        for (int j = 0; j < d_model; ++j)
        {
            output_data[i * d_model + j] += bias2_data[j];
        }
    }

    output.from_vector(output_data);

    return output;
}