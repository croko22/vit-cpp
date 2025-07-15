#ifndef MLP_H
#define MLP_H

#include "../../include/core/tensor.h"
#include "../../include/core/activation.h" // For Activation::gelu and gelu_derivative
#include "../../include/model/linear.h"
#include "../../include/model/layernorm.h"

// MLP con Layer Normalization
class MLP
{
public:
    Linear fc1, fc2;
    LayerNorm ln; // This LN is applied after the second linear layer, before the residual connection.
                  // In standard ViT, it's usually pre-norm. This structure is slightly different.
    Tensor last_hidden, last_activated;
    bool training;

    MLP(int d_model, int hidden_dim);
    Tensor forward(const Tensor &input);
    Tensor backward(const Tensor &grad_output);
    void update(float lr);
    void zero_grad();
};

#endif // MLP_H
