#ifndef TRANSFORMER_BLOCK_H
#define TRANSFORMER_BLOCK_H

#include "../../include/core/tensor.h"
#include "multihead_attention.h"
#include "mlp.h"
#include "layernorm.h"

class TransformerBlock
{
public:
    TransformerBlock(int d_model, int num_heads, int d_ff);

    Tensor forward(const Tensor &input);
    Tensor backward(const Tensor &grad_output);
    void update(float lr, int batch_size = 1);
    void zero_grad();

    MultiHeadAttention mha;
    MLP mlp;
    LayerNorm ln1, ln2;

    Tensor last_input;
    Tensor last_attn_out;
    Tensor last_residual1;
    Tensor last_normalized1;
    Tensor last_normalized2;
};

#endif