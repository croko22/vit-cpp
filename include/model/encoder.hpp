#ifndef ENCODER_LAYER_HPP
#define ENCODER_LAYER_HPP

#include "multi_head_attention.hpp"
#include "feedforward.hpp"
#include "layernorm.hpp"
#include "../core/tensor.hpp"

/**
 * @class EncoderLayer
 * @brief Un solo bloque (capa) del Encoder de un Transformer.
 * Contiene un bloque de Multi-Head Attention y un bloque de Feed-Forward,
 * cada uno seguido por una conexión residual y Layer Normalization.
 */
class EncoderLayer
{
private:
    MultiHeadAttention attention_;
    FeedForwardNetwork feed_forward_;
    LayerNormalization norm1_;
    LayerNormalization norm2_;

public:
    EncoderLayer(int d_model, int num_heads, int d_ff);
    Tensor forward(const Tensor &input);
};

#endif // ENCODER_LAYER_HPP