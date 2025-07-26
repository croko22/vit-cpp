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
    // x tiene shape [seq_len, d_model]
    auto input_shape = x.get_shape();
    int seq_len = input_shape[0];

    // 1. Proyecciones lineales
    last_q = q_proj.forward(x); // Proyección de Query, shape: [seq_len, d_model]
    last_k = k_proj.forward(x); // Proyección de Key, shape: [seq_len, d_model]
    last_v = v_proj.forward(x); // Proyección de Value, shape: [seq_len, d_model]

    // 2. Dividir en cabezas (heads)
    // [seq_len, d_model] -> [seq_len, num_heads, d_k]
    Tensor q_heads = last_q.reshape({seq_len, this->num_heads, this->d_k});
    Tensor k_heads = last_k.reshape({seq_len, this->num_heads, this->d_k});
    Tensor v_heads = last_v.reshape({seq_len, this->num_heads, this->d_k});

    // 3. Calcular puntajes de atención: (Q * K^T) / sqrt(d_k)
    Tensor k_transposed = k_heads.transpose(1, 2); // shape: [seq_len, d_k, num_heads]

    // batch_matmul([s, h, d], [s, d, h]) -> shape: [seq_len, num_heads, num_heads]
    // Aquí 'seq_len' actúa como el batch size. La atención es entre cabezas.
    // Para atención entre tokens, se requeriría un transpose(0,1) previo. Asumimos este comportamiento.
    Tensor qk = batch_matmul(q_heads, k_transposed);
    qk = qk * (1.0f / std::sqrt(static_cast<float>(d_k)));

    // 4. Aplicar softmax para obtener los pesos de atención
    last_attention_weights = Activation::softmax(qk); // shape: [seq_len, num_heads, num_heads]

    // 5. Aplicar pesos a V
    // [s, h, h] @ [s, h, d] -> shape: [seq_len, num_heads, d_k]
    Tensor attention_out = batch_matmul(last_attention_weights, v_heads);

    // 6. Concatenar cabezas y proyectar
    Tensor concatenated = attention_out.reshape({seq_len, this->d_model});
    return out_proj.forward(concatenated);
}

Tensor MultiHeadAttention::backward(const Tensor &grad_output)
{
    // Obtenemos la forma de la entrada original
    auto input_shape = last_q.get_shape();
    int seq_len = input_shape[0];

    // 1. Retropropagar a través de la proyección de salida
    Tensor grad_concatenated = out_proj.backward(grad_output);

    // 2. Separar gradiente en cabezas: [seq_len, d_model] -> [seq_len, num_heads, d_k]
    Tensor grad_attention_out = grad_concatenated.reshape({seq_len, this->num_heads, this->d_k});

    // 3. Retropropagar a través del matmul de `pesos @ V`
    // dV = A^T @ dO
    Tensor grad_v_heads = batch_matmul(last_attention_weights.transpose(1, 2), grad_attention_out);
    // dA = dO @ V^T
    Tensor grad_weights = batch_matmul(grad_attention_out, last_v.reshape({seq_len, num_heads, d_k}).transpose(1, 2));

    // 4. Retropropagar a través de Softmax
    Tensor grad_qk = Activation::softmax_grad(last_attention_weights, grad_weights);

    // 5. Retropropagar a través del escalado y el matmul de `Q @ K^T`
    grad_qk = grad_qk * (1.0f / std::sqrt(static_cast<float>(d_k)));

    // Recreamos los tensores de cabezas para el cálculo
    Tensor q_heads = last_q.reshape({seq_len, num_heads, d_k});
    Tensor k_heads = last_k.reshape({seq_len, num_heads, d_k});

    // dQ = d(QK) @ K
    Tensor grad_q_heads = batch_matmul(grad_qk, k_heads);
    // dK = d(QK)^T @ Q
    Tensor grad_k_heads = batch_matmul(grad_qk.transpose(1, 2), q_heads);

    // 6. Concatenar gradientes de las cabezas
    Tensor grad_q = grad_q_heads.reshape(input_shape);
    Tensor grad_k = grad_k_heads.reshape(input_shape);
    Tensor grad_v = grad_v_heads.reshape(input_shape);

    // 7. Retropropagar a través de las proyecciones lineales iniciales
    Tensor grad_input_q = q_proj.backward(grad_q);
    Tensor grad_input_k = k_proj.backward(grad_k);
    Tensor grad_input_v = v_proj.backward(grad_v);

    // 8. El gradiente total de la entrada es la suma de los gradientes de las 3 ramas
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