#include "../../include/model/multihead_attention.h"
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
}

Tensor MultiHeadAttention::forward(const Tensor &x)
{
    auto input_shape = x.get_shape();
    int seq_len = input_shape[0];

    last_q = q_proj.forward(x);
    last_k = k_proj.forward(x);
    last_v = v_proj.forward(x);

    Tensor q_heads = last_q.reshape({seq_len, this->num_heads, this->d_k}).transpose(0, 1);
    Tensor k_heads = last_k.reshape({seq_len, this->num_heads, this->d_k}).transpose(0, 1);
    Tensor v_heads = last_v.reshape({seq_len, this->num_heads, this->d_k}).transpose(0, 1);

    Tensor qk = batch_matmul(q_heads, k_heads.transpose(1, 2));
    qk = qk * (1.0f / std::sqrt(static_cast<float>(d_k)));

    last_attention_weights = Activation::softmax(qk);

    Tensor attention_out = batch_matmul(last_attention_weights, v_heads);

    Tensor concatenated = attention_out.transpose(0, 1).reshape({seq_len, this->d_model});
    return out_proj.forward(concatenated);
}

Tensor MultiHeadAttention::backward(const Tensor &grad_output)
{
    auto input_shape = last_q.get_shape();
    int seq_len = input_shape[0];

    Tensor grad_concatenated = out_proj.backward(grad_output);

    Tensor grad_attention_out = grad_concatenated.reshape({seq_len, num_heads, d_k}).transpose(0, 1);

    Tensor q_heads = last_q.reshape({seq_len, num_heads, d_k}).transpose(0, 1);
    Tensor k_heads = last_k.reshape({seq_len, num_heads, d_k}).transpose(0, 1);
    Tensor v_heads = last_v.reshape({seq_len, num_heads, d_k}).transpose(0, 1);

    Tensor grad_v_heads = batch_matmul(last_attention_weights.transpose(1, 2), grad_attention_out);

    Tensor grad_weights = batch_matmul(grad_attention_out, v_heads.transpose(1, 2));

    Tensor grad_qk = Activation::softmax_grad(last_attention_weights, grad_weights);

    grad_qk = grad_qk * (1.0f / std::sqrt(static_cast<float>(d_k)));

    Tensor grad_q_heads = batch_matmul(grad_qk, k_heads);

    Tensor grad_k_heads = batch_matmul(grad_qk.transpose(1, 2), q_heads);

    Tensor grad_q = grad_q_heads.transpose(0, 1).reshape(input_shape);
    Tensor grad_k = grad_k_heads.transpose(0, 1).reshape(input_shape);
    Tensor grad_v = grad_v_heads.transpose(0, 1).reshape(input_shape);

    Tensor grad_input_q = q_proj.backward(grad_q);
    Tensor grad_input_k = k_proj.backward(grad_k);
    Tensor grad_input_v = v_proj.backward(grad_v);

    return grad_input_q + grad_input_k + grad_input_v;
}

std::vector<Parameter> MultiHeadAttention::get_parameters()
{
    auto params = q_proj.get_parameters();
    auto k_params = k_proj.get_parameters();
    auto v_params = v_proj.get_parameters();
    auto out_params = out_proj.get_parameters();

    params.insert(params.end(), k_params.begin(), k_params.end());
    params.insert(params.end(), v_params.begin(), v_params.end());
    params.insert(params.end(), out_params.begin(), out_params.end());

    return params;
}

void MultiHeadAttention::zero_grad()
{
    q_proj.zero_grad();
    k_proj.zero_grad();
    v_proj.zero_grad();
    out_proj.zero_grad();
}