#include "attention.hpp"
#include <cmath>
#include <vector>
#include <algorithm>

std::vector<float> softmax(const std::vector<float> &x)
{
    std::vector<float> result(x.size());
    float max_elem = *std::max_element(x.begin(), x.end());
    float sum = 0.0f;
    for (size_t i = 0; i < x.size(); ++i)
    {
        result[i] = std::exp(x[i] - max_elem); // para estabilidad numérica
        sum += result[i];
    }
    for (size_t i = 0; i < x.size(); ++i)
    {
        result[i] /= sum;
    }
    return result;
}

MultiHeadAttention::MultiHeadAttention(int d_model, int num_heads, float dropout)
    : d_model(d_model), num_heads(num_heads), dropout(dropout)
{
    d_k = d_model / num_heads;
    // Aquí deberías inicializar W_q, W_k, W_v, W_o (matrices aleatorias)
}

std::vector<std::vector<float>> MultiHeadAttention::linear(
    const std::vector<std::vector<float>> &x, const std::vector<std::vector<float>> &W)
{
    // Producto matriz-matriz básico
    int rows = x.size(), cols = W[0].size();
    std::vector<std::vector<float>> out(rows, std::vector<float>(cols, 0));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            for (int k = 0; k < W.size(); ++k)
                out[i][j] += x[i][k] * W[k][j];
    return out;
}

std::vector<std::vector<float>> MultiHeadAttention::scaled_dot_product_attention(
    const std::vector<std::vector<float>> &q,
    const std::vector<std::vector<float>> &k,
    const std::vector<std::vector<float>> &v)
{

    int seq_len = q.size(), dim = q[0].size();
    std::vector<std::vector<float>> scores(seq_len, std::vector<float>(seq_len, 0));

    for (int i = 0; i < seq_len; ++i)
        for (int j = 0; j < seq_len; ++j)
            for (int d = 0; d < dim; ++d)
                scores[i][j] += q[i][d] * k[j][d];

    float scale = std::sqrt(dim);
    for (int i = 0; i < seq_len; ++i)
        for (int j = 0; j < seq_len; ++j)
            scores[i][j] /= scale;

    std::vector<std::vector<float>> out(seq_len, std::vector<float>(dim, 0));
    for (int i = 0; i < seq_len; ++i)
        for (int j = 0; j < seq_len; ++j)
            for (int d = 0; d < dim; ++d)
                out[i][d] += softmax(scores[i])[j] * v[j][d];
    return out;
}

std::vector<std::vector<float>> MultiHeadAttention::forward(
    const std::vector<std::vector<float>> &q,
    const std::vector<std::vector<float>> &k,
    const std::vector<std::vector<float>> &v)
{

    // Para simplificar: una sola cabeza (multihead lo añades luego)
    auto q_proj = linear(q, W_q);
    auto k_proj = linear(k, W_k);
    auto v_proj = linear(v, W_v);
    auto attn_out = scaled_dot_product_attention(q_proj, k_proj, v_proj);
    return linear(attn_out, W_o);
}
