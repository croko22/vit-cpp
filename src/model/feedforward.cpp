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
    // --- Guardar en el cache para usarlo en backward ---
    input_cache_ = input;

    // 1. Primera proyección lineal
    Tensor hidden = matmul(input, w1_) + b1_; // Asumiendo que tu '+' soporta broadcasting

    // 2. Aplicar ReLU
    Tensor activated = relu(hidden);

    // --- Guardar la salida activada en el cache ---
    hidden_activated_cache_ = activated;

    // 3. Segunda proyección lineal
    Tensor output = matmul(activated, w2_) + b2_;

    return output;
}

Tensor FeedForwardNetwork::backward(const Tensor &grad_output)
{
    // Ahora esta función funcionará porque las variables de cache existen

    // 1. Backward a través de la segunda capa lineal (W2, b2)
    auto [grad_activated, grad_w2] = matmul_backward(grad_output, hidden_activated_cache_, w2_);

    Tensor grad_b2 = sum(grad_output, 0, true);
    // Acumular gradientes
    if (b2_.grad_)
        *(b2_.grad_) = *(b2_.grad_) + grad_b2;
    if (w2_.grad_)
        *(w2_.grad_) = *(w2_.grad_) + grad_w2;

    // 2. Backward a través de ReLU
    Tensor grad_hidden = relu_backward(grad_activated, hidden_activated_cache_);

    // 3. Backward a través de la primera capa lineal (W1, b1)
    // ¡Esta línea ya no dará error!
    auto [grad_input, grad_w1] = matmul_backward(grad_hidden, input_cache_, w1_);

    Tensor grad_b1 = sum(grad_hidden, 0, true);
    if (b1_.grad_)
        *(b1_.grad_) = *(b1_.grad_) + grad_b1;
    if (w1_.grad_)
        *(w1_.grad_) = *(w1_.grad_) + grad_w1;

    return grad_input;
}

void FeedForwardNetwork::get_parameters(std::vector<Tensor *> &params)
{
    params.push_back(&w1_);
    params.push_back(&b1_);
    params.push_back(&w2_);
    params.push_back(&b2_);
}