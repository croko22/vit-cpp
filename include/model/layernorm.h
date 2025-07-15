#ifndef LAYERNORM_H
#define LAYERNORM_H

#include "../../include/core/tensor.h"
#include <cmath>
#include <algorithm>

class LayerNorm
{
public:
    Tensor gamma, beta;
    Tensor gamma_grad, beta_grad;
    Tensor last_input, last_mean, last_var;
    int d_model;
    float eps;
    LayerNorm(int d_mod);
    Tensor forward(const Tensor &input);
    Tensor backward(const Tensor &grad_output);
    void update(float lr);
    void zero_grad();
};

#endif // LAYERNORM_H