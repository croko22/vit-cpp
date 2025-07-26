#include "../../include/model/multihead_attention.h" // Ajusta tu ruta
#include "../../include/core/activation.h"
#include <cmath>
#include <vector>

MultiHeadAttention::MultiHeadAttention(int d_model, int num_heads)
    : d_model(d_model),
      num_heads(num_heads),
      d_k(d_model / num_heads),
      q_proj(d_model, d_model),
      k_proj(d_model, d_model),
      v_proj(d_model, d_model),
      out_proj(d_model, d_model)
{
    // El constructor ya está bien, no necesita cambios.
}

Tensor MultiHeadAttention::forward(const Tensor &x)
{
    auto input_shape = x.get_shape();
    int seq_len = input_shape[0];

    // 1. Proyecciones lineales
    last_q = q_proj.forward(x);
    last_k = k_proj.forward(x);
    last_v = v_proj.forward(x);

    // 2. Dividir en cabezas y permutar para el batch matmul
    // [seq_len, d_model] -> [seq_len, num_heads, d_k] -> [num_heads, seq_len, d_k]
    Tensor q_heads = last_q.reshape({seq_len, this->num_heads, this->d_k}).transpose(0, 1);
    Tensor k_heads = last_k.reshape({seq_len, this->num_heads, this->d_k}).transpose(0, 1);
    Tensor v_heads = last_v.reshape({seq_len, this->num_heads, this->d_k}).transpose(0, 1);

    // 3. Calcular puntajes de atención: (Q * K^T) / sqrt(d_k)
    // QK = [h, s, d] @ [h, d, s] -> [h, s, s]
    Tensor qk = batch_matmul(q_heads, k_heads.transpose(1, 2));
    qk = qk * (1.0f / std::sqrt(static_cast<float>(d_k)));

    // 4. Aplicar softmax
    last_attention_weights = Activation::softmax(qk); // Shape: [num_heads, seq_len, seq_len]

    // 5. Aplicar pesos a V
    // out = [h, s, s] @ [h, s, d] -> [h, s, d]
    Tensor attention_out = batch_matmul(last_attention_weights, v_heads);

    // 6. Concatenar cabezas y proyectar
    // [h, s, d] -> [s, h, d] -> [s, d_model]
    Tensor concatenated = attention_out.transpose(0, 1).reshape({seq_len, this->d_model});
    return out_proj.forward(concatenated);
}

Tensor MultiHeadAttention::backward(const Tensor &grad_output)
{
    auto input_shape = last_q.get_shape();
    int seq_len = input_shape[0];

    // 1. Backprop a través de la proyección de salida
    Tensor grad_concatenated = out_proj.backward(grad_output);

    // 2. Deshacer la concatenación y la transposición
    // [s, d_model] -> [s, h, d_k] -> [h, s, d_k]
    Tensor grad_attention_out = grad_concatenated.reshape({seq_len, num_heads, d_k}).transpose(0, 1);

    // Recrear tensores del forward pass con las formas correctas para el backward
    Tensor q_heads = last_q.reshape({seq_len, num_heads, d_k}).transpose(0, 1);
    Tensor k_heads = last_k.reshape({seq_len, num_heads, d_k}).transpose(0, 1);
    Tensor v_heads = last_v.reshape({seq_len, num_heads, d_k}).transpose(0, 1);

    // 3. Backprop a través de `pesos @ V`
    // d(V) = A^T @ d(out)
    Tensor grad_v_heads = batch_matmul(last_attention_weights.transpose(1, 2), grad_attention_out);
    // d(A) = d(out) @ V^T
    Tensor grad_weights = batch_matmul(grad_attention_out, v_heads.transpose(1, 2));

    // 4. Backprop a través de Softmax
    Tensor grad_qk = Activation::softmax_grad(last_attention_weights, grad_weights);

    // 5. Backprop a través del escalado
    grad_qk = grad_qk * (1.0f / std::sqrt(static_cast<float>(d_k)));

    // 6. Backprop a través de `Q @ K^T`
    // d(Q) = d(QK) @ K
    Tensor grad_q_heads = batch_matmul(grad_qk, k_heads);
    // d(K) = d(QK)^T @ Q
    Tensor grad_k_heads = batch_matmul(grad_qk.transpose(1, 2), q_heads);

    // 7. Deshacer la transposición y el reshape de los gradientes de las cabezas
    Tensor grad_q = grad_q_heads.transpose(0, 1).reshape(input_shape);
    Tensor grad_k = grad_k_heads.transpose(0, 1).reshape(input_shape);
    Tensor grad_v = grad_v_heads.transpose(0, 1).reshape(input_shape);

    // 8. Backprop a través de las proyecciones lineales iniciales
    Tensor grad_input_q = q_proj.backward(grad_q);
    Tensor grad_input_k = k_proj.backward(grad_k);
    Tensor grad_input_v = v_proj.backward(grad_v);

    // 9. Sumar gradientes
    return grad_input_q + grad_input_k + grad_input_v;
}

void MultiHeadAttention::update(float lr, int batch_size)
{
    q_proj.update(lr, batch_size);
    k_proj.update(lr, batch_size);
    v_proj.update(lr, batch_size);
    out_proj.update(lr, batch_size);
}

void MultiHeadAttention::zero_grad()
{
    q_proj.zero_grad();
    k_proj.zero_grad();
    v_proj.zero_grad();
    out_proj.zero_grad();
}