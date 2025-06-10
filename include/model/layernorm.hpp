#pragma once
#include <vector>

class LayerNorm
{
public:
    LayerNorm(int dim);
    std::vector<std::vector<float>> forward(
        const std::vector<std::vector<float>> &x,
        const std::vector<std::vector<float>> &residual);

private:
    int dim;
};
