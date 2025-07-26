#ifndef LAYERNORM_H
#define LAYERNORM_H

#include "../../include/core/tensor.h"
#include <cmath>
#include <algorithm>

class LayerNorm
{
public:
    int d_model;
    float eps;

    // 2. Parámetros aprendibles y sus gradientes
    Tensor gamma, beta;
    Tensor gamma_grad, beta_grad;

    // 3. Tensores cacheados para backprop
    Tensor last_input, last_mean, last_var;
    LayerNorm(int d_mod);
    Tensor forward(const Tensor &input);
    Tensor backward(const Tensor &grad_output);
    void update(float lr, int batch_size = 1);
    void zero_grad();
};

#endif // LAYERNORM_H