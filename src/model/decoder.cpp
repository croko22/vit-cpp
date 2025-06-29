#include "../../include/model/decoder.hpp"

DecoderLayer::DecoderLayer(int d_model, int num_heads, int d_ff)
    : masked_self_attention_(d_model, num_heads),
      cross_attention_(d_model, num_heads),
      feed_forward_(d_model, d_ff),
      norm1_(d_model),
      norm2_(d_model),
      norm3_(d_model)
{
}

Tensor DecoderLayer::forward(const Tensor &target_input, const Tensor &encoder_output, const Tensor *look_ahead_mask, const Tensor *padding_mask)
{
    // 1. Bloque de Masked Multi-Head Self-Attention
    // Q, K, V vienen de la misma entrada del decoder (target_input).
    // Usamos la look_ahead_mask para no ver el futuro.
    Tensor self_attn_output = masked_self_attention_.forward(target_input, target_input, target_input, look_ahead_mask);

    // 2. Add & Norm 1
    Tensor sublayer1_output = norm1_.forward(target_input + self_attn_output);

    // 3. Bloque de Cross-Attention (Encoder-Decoder Attention)
    // Query (Q) viene de la salida del bloque anterior del decoder (sublayer1_output).
    // Key (K) y Value (V) vienen de la salida del Encoder (encoder_output).
    // Esto es CRUCIAL: aquí el decoder "consulta" la información de la oración de entrada.
    Tensor cross_attn_output = cross_attention_.forward(sublayer1_output, encoder_output, encoder_output, padding_mask);

    // 4. Add & Norm 2
    Tensor sublayer2_output = norm2_.forward(sublayer1_output + cross_attn_output);

    // 5. Bloque Feed-Forward
    Tensor ffn_output = feed_forward_.forward(sublayer2_output);

    // 6. Add & Norm 3
    Tensor output = norm3_.forward(sublayer2_output + ffn_output);

    return output;
}