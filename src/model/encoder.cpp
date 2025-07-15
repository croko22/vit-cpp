#include "../../include/model/encoder.h"
#include <iostream>

TransformerBlock::TransformerBlock(int d_model)
    : attention_proj(d_model, d_model), mlp(d_model, d_model * 2),
      ln1(d_model), ln2(d_model)
{
}

Tensor TransformerBlock::forward(const Tensor &input)
{
    last_input = input;

    last_normalized1 = ln1.forward(input);
    Tensor attn_out = attention_proj.forward(last_normalized1);
    last_attn_out = attn_out;

    last_residual1 = input + attn_out;

    last_normalized2 = ln2.forward(last_residual1);
    Tensor mlp_out = mlp.forward(last_normalized2);

    return last_residual1 + mlp_out;
}

Tensor TransformerBlock::backward(const Tensor &grad_output)
{

    Tensor grad_residual1_from_mlp = grad_output;
    Tensor grad_mlp_out = grad_output;

    Tensor grad_normalized2 = mlp.backward(grad_mlp_out);
    Tensor grad_residual1_from_ln2 = ln2.backward(grad_normalized2);

    Tensor grad_residual1 = grad_residual1_from_mlp + grad_residual1_from_ln2;

    Tensor grad_input_from_attn = grad_residual1;
    Tensor grad_input_direct = grad_residual1;

    Tensor grad_normalized1 = attention_proj.backward(grad_input_from_attn);
    Tensor grad_input_from_ln1 = ln1.backward(grad_normalized1);

    Tensor grad_input = grad_input_direct + grad_input_from_ln1;

    return grad_input;
}

void TransformerBlock::update(float lr)
{
    attention_proj.update(lr);
    mlp.update(lr);
    ln1.update(lr);
    ln2.update(lr);
}

void TransformerBlock::zero_grad()
{
    attention_proj.zero_grad();
    mlp.zero_grad();
    ln1.zero_grad();
    ln2.zero_grad();
}
