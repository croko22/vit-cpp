#include "layernorm.hpp"
#include <cmath>

LayerNorm::LayerNorm(int dim) : dim(dim) {}

std::vector<std::vector<float>> LayerNorm::forward(
    const std::vector<std::vector<float>> &x,
    const std::vector<std::vector<float>> &residual)
{

    std::vector<std::vector<float>> out = x;
    for (int i = 0; i < x.size(); ++i)
    {
        std::vector<float> temp(dim);
        for (int d = 0; d < dim; ++d)
            temp[d] = x[i][d] + residual[i][d];

        float mean = 0, var = 0;
        for (float v : temp)
            mean += v;
        mean /= dim;
        for (float v : temp)
            var += (v - mean) * (v - mean);
        var /= dim;
        float std = std::sqrt(var + 1e-5);

        for (int d = 0; d < dim; ++d)
            out[i][d] = (temp[d] - mean) / std;
    }
    return out;
}
