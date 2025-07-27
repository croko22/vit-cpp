#ifndef MLP_H
#define MLP_H

#include "../../include/core/tensor.h"
#include "../../include/core/activation.h"
#include "../../include/model/linear.h"
#include "../../include/model/layernorm.h"

class MLP
{
public:
    Linear fc1, fc2;
    LayerNorm ln;

    Tensor last_hidden, last_activated;
    Tensor dropout_mask;
    bool training;
    float drop_prob;

    MLP(int d_model, int hidden_dim, float drop_prob = 0.1);
    Tensor forward(const Tensor &input);
    Tensor backward(const Tensor &grad_output);
    void update(float lr, int batch_size = 1);
    void zero_grad();
};

#endif
