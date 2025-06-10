#include "../../include/model/encoder.hpp"

TransformerEncoder::TransformerEncoder(int num_layers, int d_model, int num_heads, int d_ff, float dropout)
    : num_layers(num_layers)
{
    for (int i = 0; i < num_layers; ++i)
    {
        self_attn_layers.emplace_back(d_model, num_heads, dropout);
        ff_layers.emplace_back(d_model, d_ff, dropout);
        norm1_layers.emplace_back(d_model);
        norm2_layers.emplace_back(d_model);
    }
}

std::vector<std::vector<float>> TransformerEncoder::forward(const std::vector<std::vector<float>> &input)
{
    auto x = input;
    for (int i = 0; i < num_layers; ++i)
    {
        auto attn_out = self_attn_layers[i].forward(x, x, x);  // self-attention
        auto norm1_out = norm1_layers[i].forward(x, attn_out); // residual + norm
        auto ff_out = ff_layers[i].forward(norm1_out);
        x = norm2_layers[i].forward(norm1_out, ff_out); // residual + norm
    }
    return x;
}
