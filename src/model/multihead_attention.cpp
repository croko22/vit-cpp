#include "../../include/model/multihead_attention.h"
#include "../../include/core/activation.h"
#include <cmath>

MultiHeadAttention::MultiHeadAttention(int d_model, int num_heads) 
    : d_model(d_model),
      num_heads(num_heads),
      d_k(d_model / num_heads),
      q_proj(d_model, d_model),
      k_proj(d_model, d_model), 
      v_proj(d_model, d_model),
      out_proj(d_model, d_model)
{
}

Tensor MultiHeadAttention::forward(const Tensor &x)
{
    // Proyectar Q, K, V
    last_q = q_proj.forward(x);
    last_k = k_proj.forward(x);
    last_v = v_proj.forward(x);

    // Dividir en cabezas (reshape)
    int seq_len = x.rows;
    Tensor q_heads = last_q.reshape(seq_len, num_heads, d_k); // [seq_len, num_heads, d_k]
    Tensor k_heads = last_k.reshape(seq_len, num_heads, d_k);
    Tensor v_heads = last_v.reshape(seq_len, num_heads, d_k);

    // Calcular atención (Q * K^T / sqrt(d_k))
    Tensor qk = q_heads.batch_matmul(k_heads.transpose(1, 2)); // [seq_len, num_heads, seq_len]
    qk = qk * (1.0f / std::sqrt(d_k));

    // Softmax para obtener pesos
    last_attention_weights = Activation::softmax(qk); // [seq_len, num_heads, seq_len]

    // Aplicar a valores (V)
    Tensor attention_out = last_attention_weights.batch_matmul(v_heads); // [seq_len, num_heads, d_k]

    // Concatenar cabezas y proyectar
    Tensor concatenated = attention_out.reshape(seq_len, d_model); // [seq_len, d_model]
    return out_proj.forward(concatenated);
}

Tensor MultiHeadAttention::backward(const Tensor &grad_output) {
    // 1. Gradiente de out_proj
    Tensor grad_concatenated = out_proj.backward(grad_output);

    // 2. Reshape para separar cabezas
    int seq_len = grad_output.rows;
    Tensor grad_attention_out = grad_concatenated.reshape(seq_len, num_heads, d_k);

    // 3. Gradiente de atención (softmax + matmul)
    Tensor grad_v_heads = last_attention_weights.transpose(1, 2).batch_matmul(grad_attention_out);
    Tensor grad_weights = grad_attention_out.batch_matmul(last_v.transpose(1, 2));

    // 4. Gradiente de softmax
    Tensor grad_qk = Activation::softmax_grad(last_attention_weights, grad_weights);

    // 5. Gradiente de Q, K (incluyendo escalado)
    grad_qk = grad_qk * (1.0f / std::sqrt(d_k));
    Tensor grad_q_heads = grad_qk.batch_matmul(last_k);
    Tensor grad_k_heads = grad_qk.transpose(1, 2).batch_matmul(last_q);

    // 6. Reshape y backprop a proyecciones
    Tensor grad_q = grad_q_heads.reshape(seq_len, d_model);
    Tensor grad_k = grad_k_heads.reshape(seq_len, d_model);
    Tensor grad_v = grad_v_heads.reshape(seq_len, d_model);

    q_proj.backward(grad_q);
    k_proj.backward(grad_k);
    v_proj.backward(grad_v);

    // Gradiente total (suma de grad_q + grad_k + grad_v)
    return grad_q + grad_k + grad_v;
}

void MultiHeadAttention::update(float lr) {
    q_proj.update(lr);
    k_proj.update(lr);
    v_proj.update(lr);
    out_proj.update(lr);
}

void MultiHeadAttention::zero_grad() {
    q_proj.zero_grad();
    k_proj.zero_grad();
    v_proj.zero_grad();
    out_proj.zero_grad();
}