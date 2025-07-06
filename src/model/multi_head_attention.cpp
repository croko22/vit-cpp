#include "../../include/model/multi_head_attention.hpp"
#include "../../include/core/ops.hpp"
#include <cmath>
#include <iostream>
#include <random>

MultiHeadAttention::MultiHeadAttention(int d_model, int num_heads)
    : w_q_({d_model, d_model}),
      w_k_({d_model, d_model}),
      w_v_({d_model, d_model}),
      w_o_({d_model, d_model}),
      d_model_(d_model),
      num_heads_(num_heads)
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
    // --- CACHEAR LAS ENTRADAS ORIGINALES ---
    query_input_ = query;
    key_input_ = key;
    value_input_ = value;

    // 1. Proyección lineal: Q' = QWq, K' = KWk, V' = VWv
    // Guardamos los resultados en las variables de la clase para usarlos en backward.
    q_proj_ = matmul(query_input_, w_q_);
    k_proj_ = matmul(key_input_, w_k_);
    v_proj_ = matmul(value_input_, w_v_);

    // 2. Puntuaciones de atención (Attention Scores)
    Tensor scores = matmul(q_proj_, k_proj_.transpose());
    Tensor scaled_scores = scores * scale_;

    // 3. Pesos de atención (Attention Weights)
    // Se aplica softmax para obtener una distribución de probabilidad.
    attention_weights_ = softmax(scaled_scores);

    // 4. Calcular el contexto (Context Vector)
    // Se pondera cada vector de valor (V) según los pesos de atención.
    context_ = matmul(attention_weights_, v_proj_);

    // 5. Proyección de salida final
    return matmul(context_, w_o_);
}

std::tuple<Tensor, Tensor, Tensor> MultiHeadAttention::backward(const Tensor &grad_output)
{
    // El backward pass sigue la regla de la cadena en orden inverso al forward pass.

    // 1. Backward a través de la proyección de salida (W_o)
    auto [grad_context, grad_w_o] = matmul_backward(grad_output, context_, w_o_);
    *(w_o_.grad_) = grad_w_o; // Actualizar el gradiente del peso W_o

    // 2. Backward a través de matmul(attention_weights, V')
    auto [grad_attention_weights, grad_v_proj] = matmul_backward(grad_context, attention_weights_, v_proj_);

    // 3. Backward a través de Softmax
    Tensor grad_scaled_scores = softmax_backward(grad_attention_weights, attention_weights_);

    // 4. Backward a través del escalamiento (operación elemental)
    Tensor grad_scores = grad_scaled_scores * scale_;

    // 5. Backward a través de matmul(Q', K'.T)
    auto [grad_q_proj, grad_k_proj_T] = matmul_backward(grad_scores, q_proj_, k_proj_.transpose());
    Tensor grad_k_proj = grad_k_proj_T.transpose();

    // 6. Propagar gradientes a través de las proyecciones iniciales (W_q, W_k, W_v)
    // Para cada proyección, calculamos el gradiente del peso y el gradiente de la entrada.
    auto [grad_query, grad_w_q] = matmul_backward(grad_q_proj, query_input_, w_q_);
    *(w_q_.grad_) = grad_w_q;

    auto [grad_key, grad_w_k] = matmul_backward(grad_k_proj, key_input_, w_k_);
    *(w_k_.grad_) = grad_w_k;

    // El grad_v_proj ya lo calculamos en el paso 2. Ahora lo propagamos a través de W_v.
    auto [grad_value, grad_w_v] = matmul_backward(grad_v_proj, value_input_, w_v_);
    *(w_v_.grad_) = grad_w_v;

    // Devolvemos los gradientes con respecto a las entradas originales (query, key, value).
    // En self-attention (encoder), estos tres gradientes se sumarían, ya que la entrada es la misma.
    // En cross-attention (decoder), grad_query iría al sub-bloque anterior del decoder,
    // mientras que grad_key y grad_value irían hacia la salida del encoder.
    return {grad_query, grad_key, grad_value};
}

void MultiHeadAttention::zero_all_grads()
{
    if (w_q_.grad_)
        w_q_.grad_->zero_grad();
    if (w_k_.grad_)
        w_k_.grad_->zero_grad();
    if (w_v_.grad_)
        w_v_.grad_->zero_grad();
    if (w_o_.grad_)
        w_o_.grad_->zero_grad();
}

// En la clase MultiHeadAttention
void MultiHeadAttention::get_parameters(std::vector<Tensor *> &params)
{
    params.push_back(&w_q_);
    params.push_back(&w_k_);
    params.push_back(&w_v_);
    params.push_back(&w_o_);
}