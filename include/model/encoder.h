#ifndef TRANSFORMER_BLOCK_H
#define TRANSFORMER_BLOCK_H

#include "../../include/core/tensor.h"
#include "linear.h"
#include "layernorm.h"
#include "mlp.h"  // TransformerBlock uses MLP
#include <memory> // For std::unique_ptr

// Transformer Block mejorado
class TransformerBlock
{
public:
    Linear attention_proj; // Simplified attention: just a linear projection
    MLP mlp;
    LayerNorm ln1, ln2; // Pre-norm layers
    Tensor last_input, last_attn_out, last_residual1, last_normalized1, last_normalized2;

    TransformerBlock(int d_model);
    Tensor forward(const Tensor &input);
    Tensor backward(const Tensor &grad_output);
    void update(float lr);
    void zero_grad();
};

#endif // TRANSFORMER_BLOCK_H
