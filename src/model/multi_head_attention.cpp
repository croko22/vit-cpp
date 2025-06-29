#include "../../include/model/multi_head_attention.hpp"
#include "../../include/core/ops.hpp"
#include <cmath>
#include <iostream>
#include <random>

MultiHeadAttention::MultiHeadAttention(int d_model, int num_heads)
    : d_model_(d_model),
      num_heads_(num_heads),
      w_q_({d_model, d_model}),
      w_k_({d_model, d_model}),
      w_v_({d_model, d_model}),
      w_o_({d_model, d_model})
{
    int d_k = d_model / num_heads;
    scale_ = 1.0f / std::sqrt(static_cast<float>(d_k));

    // INICIALIZAR LAS MATRICES DE PESOS
    std::mt19937 gen(std::random_device{}());

    // Xavier/Glorot initialization
    float xavier_std = std::sqrt(1.0f / d_model);
    std::normal_distribution<float> dist(0.0f, xavier_std);

    // Inicializar W_q
    std::vector<float> wq_data(d_model * d_model);
    for (auto &val : wq_data)
        val = dist(gen);
    w_q_.from_vector(wq_data);

    // Inicializar W_k
    std::vector<float> wk_data(d_model * d_model);
    for (auto &val : wk_data)
        val = dist(gen);
    w_k_.from_vector(wk_data);

    // Inicializar W_v
    std::vector<float> wv_data(d_model * d_model);
    for (auto &val : wv_data)
        val = dist(gen);
    w_v_.from_vector(wv_data);

    // Inicializar W_o
    std::vector<float> wo_data(d_model * d_model);
    for (auto &val : wo_data)
        val = dist(gen);
    w_o_.from_vector(wo_data);

    std::cout << "MultiHeadAttention initialized: d_model=" << d_model
              << ", num_heads=" << num_heads << ", d_k=" << d_k << std::endl;
}

Tensor MultiHeadAttention::forward(const Tensor &query, const Tensor &key, const Tensor &value, const Tensor *mask)
{
    // 1. Proyección lineal a Q, K, V
    Tensor q_proj = matmul(query, w_q_);
    Tensor k_proj = matmul(key, w_k_);
    Tensor v_proj = matmul(value, w_v_);

    // NOTA: La lógica de dividir en cabezales iría aquí.

    // 2. Scaled Dot-Product Attention
    Tensor scores = matmul(q_proj, k_proj.transpose());
    Tensor scaled_scores = scores * scale_;

    // 3. Aplicar máscara (si existe)
    // La máscara se suma. Donde la máscara es un número muy negativo,
    // el resultado del softmax será cero.
    if (mask)
    {
        // Asumimos que el operador `+` de Tensor soporta broadcasting o tiene la misma forma.
        scaled_scores = scaled_scores + *mask;
    }

    Tensor attention_weights = softmax(scaled_scores);
    Tensor context = matmul(attention_weights, v_proj);

    // NOTA: La lógica de concatenar cabezales iría aquí.

    // 4. Proyección lineal de salida
    Tensor output = matmul(context, w_o_);

    return output;
}