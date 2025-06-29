#ifndef DECODER_HPP
#define DECODER_HPP

#include "multi_head_attention.hpp"
#include "feedforward.hpp"
#include "layernorm.hpp"
#include "../core/tensor.hpp"

/**
 * @class DecoderLayer
 * @brief Un solo bloque (capa) del Decoder de un Transformer.
 * Contiene Masked Self-Attention, Cross-Attention, y una red Feed-Forward.
 */
class DecoderLayer
{
private:
    MultiHeadAttention masked_self_attention_;
    MultiHeadAttention cross_attention_;
    FeedForwardNetwork feed_forward_;

    // El decoder necesita 3 capas de normalización
    LayerNormalization norm1_;
    LayerNormalization norm2_;
    LayerNormalization norm3_;

public:
    DecoderLayer(int d_model, int num_heads, int d_ff);

    /**
     * @brief Realiza el paso forward de la capa del decoder.
     * @param target_input El tensor de entrada del target (la secuencia generada hasta ahora).
     * @param encoder_output El tensor de salida de todo el stack de encoders.
     * @param look_ahead_mask Máscara para el self-attention, para evitar ver el futuro.
     * @param padding_mask Máscara para el cross-attention, para ignorar el padding (opcional).
     * @return El tensor de salida de la capa.
     */
    Tensor forward(const Tensor &target_input, const Tensor &encoder_output, const Tensor *look_ahead_mask, const Tensor *padding_mask = nullptr);
};

#endif // DECODER_HPP