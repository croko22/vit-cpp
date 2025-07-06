#ifndef ENCODER_LAYER_HPP
#define ENCODER_LAYER_HPP

#include "multi_head_attention.hpp"
#include "feedforward.hpp"
#include "layernorm.hpp"
#include "../core/tensor.hpp"

class EncoderLayer
{
private:
    MultiHeadAttention attention_;
    FeedForwardNetwork feed_forward_;
    LayerNormalization norm1_;
    LayerNormalization norm2_;

    // --- Cache para el Backward Pass ---
    // Guardamos los tensores necesarios del forward para usarlos en el backward.
    Tensor input_cache_;
    Tensor sublayer1_output_cache_;

public:
    EncoderLayer(int d_model, int num_heads, int d_ff);
    Tensor forward(const Tensor &input);
    Tensor backward(const Tensor &grad_output);
    void get_parameters(std::vector<Tensor *> &params);
};

#endif // ENCODER_LAYER_HPP