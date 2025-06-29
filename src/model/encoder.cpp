#include "../../include/model/encoder.hpp"

EncoderLayer::EncoderLayer(int d_model, int num_heads, int d_ff)
    : attention_(d_model, num_heads),
      feed_forward_(d_model, d_ff),
      norm1_(d_model),
      norm2_(d_model)
{
}

Tensor EncoderLayer::forward(const Tensor &input)
{
    // 1. Bloque de Multi-Head Attention
    Tensor attn_output = attention_.forward(input, input, input, nullptr);

    // 2. Conexión Residual (Add) y Layer Normalization (Norm)
    Tensor sublayer1_output = norm1_.forward(input + attn_output);

    // 3. Bloque Feed-Forward
    Tensor ffn_output = feed_forward_.forward(sublayer1_output);

    // 4. Segunda Conexión Residual (Add) y Layer Normalization (Norm)
    Tensor output = norm2_.forward(sublayer1_output + ffn_output);

    return output;
}