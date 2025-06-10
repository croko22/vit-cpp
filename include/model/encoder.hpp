#pragma once
#include <vector>
#include "attention.hpp"
#include "layernorm.hpp"
#include "feedforward.hpp"

class TransformerEncoder
{
public:
    TransformerEncoder(int num_layers, int d_model, int num_heads, int d_ff, float dropout);
    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>> &input);

private:
    int num_layers;
    std::vector<MultiHeadAttention> self_attn_layers;
    std::vector<FeedForwardNetwork> ff_layers;
    std::vector<LayerNorm> norm1_layers;
    std::vector<LayerNorm> norm2_layers;
};
