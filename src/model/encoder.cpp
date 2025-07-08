#include "../../include/model/encoder.hpp"
#include <iostream>

EncoderLayer::EncoderLayer(int d_model, int num_heads, int d_ff)
    : attention_(d_model, num_heads),
      feed_forward_(d_model, d_ff),
      norm1_(d_model),
      norm2_(d_model)
{
}

Tensor EncoderLayer::forward(const Tensor &input)
{
    // Guardamos el input para usarlo en la conexión residual del backward
    input_cache_ = input;

    // 1. Bloque de Multi-Head Attention
    Tensor attn_output = attention_.forward(input, input, input, nullptr);

    // 2. Conexión Residual (Add) y Layer Normalization (Norm)
    Tensor sublayer1_output = norm1_.forward(input + attn_output);

    // Guardamos la salida de la primera subcapa para el backward
    sublayer1_output_cache_ = sublayer1_output;

    // 3. Bloque Feed-Forward
    Tensor ffn_output = feed_forward_.forward(sublayer1_output);

    // 4. Segunda Conexión Residual (Add) y Layer Normalization (Norm)
    Tensor output = norm2_.forward(sublayer1_output + ffn_output);

    return output;
}

Tensor EncoderLayer::backward(const Tensor &grad_output)
{
    std::cout << "[Encoder::backward] grad_output: ";
    grad_output.print("grad_output");

    Tensor grad_from_norm2 = norm2_.backward(grad_output);
    grad_from_norm2.print("grad_from_norm2");

    Tensor grad_sublayer1_from_residual2 = grad_from_norm2;
    Tensor grad_ffn_output = grad_from_norm2;

    Tensor grad_sublayer1_from_ffn = feed_forward_.backward(grad_ffn_output);
    grad_sublayer1_from_ffn.print("grad_sublayer1_from_ffn");

    Tensor grad_sublayer1_total = grad_sublayer1_from_residual2 + grad_sublayer1_from_ffn;
    grad_sublayer1_total.print("grad_sublayer1_total");

    Tensor grad_from_norm1 = norm1_.backward(grad_sublayer1_total);
    grad_from_norm1.print("grad_from_norm1");

    Tensor grad_input_from_residual1 = grad_from_norm1;
    Tensor grad_attn_output = grad_from_norm1;

    std::cout << "[Encoder::backward] -> attention_.backward..." << std::endl;

    auto [grad_q, grad_k, grad_v] = attention_.backward(grad_attn_output);
    grad_q.print("grad_q");
    grad_k.print("grad_k");
    grad_v.print("grad_v");

    Tensor grad_input_from_attn = grad_q + grad_k + grad_v;
    grad_input_from_attn.print("grad_input_from_attn");

    Tensor grad_input = grad_input_from_residual1 + grad_input_from_attn;
    grad_input.print("grad_input");

    return grad_input;
}

void EncoderLayer::get_parameters(std::vector<Tensor *> &params)
{
    attention_.get_parameters(params);
    feed_forward_.get_parameters(params);
    norm1_.get_parameters(params);
    norm2_.get_parameters(params);
}