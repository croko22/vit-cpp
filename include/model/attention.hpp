#pragma once
#include <vector>

class MultiHeadAttention
{
public:
    MultiHeadAttention(int d_model, int num_heads, float dropout);
    std::vector<std::vector<float>> forward(
        const std::vector<std::vector<float>> &q,
        const std::vector<std::vector<float>> &k,
        const std::vector<std::vector<float>> &v);

private:
    int d_model, num_heads, d_k;
    float dropout;
    std::vector<std::vector<float>> W_q, W_k, W_v, W_o;

    std::vector<std::vector<float>> linear(const std::vector<std::vector<float>> &x, const std::vector<std::vector<float>> &W);
    std::vector<std::vector<float>> scaled_dot_product_attention(
        const std::vector<std::vector<float>> &q,
        const std::vector<std::vector<float>> &k,
        const std::vector<std::vector<float>> &v);
};
