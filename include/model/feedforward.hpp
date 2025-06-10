#pragma once
#include <vector>

class FeedForwardNetwork
{
public:
    FeedForwardNetwork(int d_model, int d_ff, float dropout);
    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>> &x);

private:
    std::vector<std::vector<float>> W1, W2;
    std::vector<float> relu(const std::vector<float> &x);
    std::vector<std::vector<float>> linear(const std::vector<std::vector<float>> &x, const std::vector<std::vector<float>> &W);
};
