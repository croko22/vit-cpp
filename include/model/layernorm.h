#ifndef LAYERNORM_H
#define LAYERNORM_H

#include "../../include/core/tensor.h"
#include "../../include/core/optimizer.h"
#include <cmath>
#include <algorithm>

class LayerNorm
{
public:
    int d_model;
    float eps;

    Tensor gamma, beta;
    Tensor gamma_grad, beta_grad;

    Tensor last_input, last_mean, last_var;
    LayerNorm(int d_mod);
    Tensor forward(const Tensor &input);
    Tensor backward(const Tensor &grad_output);

    void zero_grad();
    std::vector<Parameter> get_parameters();
};

#endif