#ifndef LAYERNORM_HPP
#define LAYERNORM_HPP

#include "../core/tensor.hpp"

class LayerNormalization
{
private:
    int feature_size_;
    float epsilon_;

    // Parámetros aprendibles
    Tensor gamma_; // scale
    Tensor beta_;  // shift

public:
    LayerNormalization(int feature_size, float epsilon = 1e-5);
    Tensor forward(const Tensor &input);
};

#endif // LAYERNORM_HPP