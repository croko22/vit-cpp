#include "../../include/model/feedforward.hpp"
#include <random>
#include <algorithm>

FeedForwardNetwork::FeedForwardNetwork(int d_model, int d_ff, float dropout)
{
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    W1.resize(d_model, std::vector<float>(d_ff));
    W2.resize(d_ff, std::vector<float>(d_model));

    for (auto &row : W1)
        for (auto &val : row)
            val = dist(gen);

    for (auto &row : W2)
        for (auto &val : row)
            val = dist(gen);
}

std::vector<float> FeedForwardNetwork::relu(const std::vector<float> &x)
{
    std::vector<float> out(x.size());
    for (int i = 0; i < x.size(); ++i)
        out[i] = std::max(0.0f, x[i]);
    return out;
}

std::vector<std::vector<float>> FeedForwardNetwork::linear(
    const std::vector<std::vector<float>> &x, const std::vector<std::vector<float>> &W)
{
    int rows = x.size(), cols = W[0].size();
    std::vector<std::vector<float>> out(rows, std::vector<float>(cols, 0));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            for (int k = 0; k < W.size(); ++k)
                out[i][j] += x[i][k] * W[k][j];
    return out;
}

std::vector<std::vector<float>> FeedForwardNetwork::forward(const std::vector<std::vector<float>> &x)
{
    auto h = linear(x, W1);
    for (auto &row : h)
        row = relu(row);
    return linear(h, W2);
}
