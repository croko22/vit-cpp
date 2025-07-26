#include "../../include/model/encoder.h"

TransformerBlock::TransformerBlock(int d_model, int num_heads, int d_ff)
    : mha(d_model, num_heads),
      mlp(d_model, d_ff),
      ln1(d_model),
      ln2(d_model)
{
}

Tensor TransformerBlock::forward(const Tensor &input)
{

    last_input = input;

    last_normalized1 = ln1.forward(input);

    last_attn_out = mha.forward(last_normalized1);

    last_residual1 = input + last_attn_out;

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

    Tensor grad_input_direct = grad_residual1;
    Tensor grad_attn_out = grad_residual1;

    Tensor grad_normalized1 = mha.backward(grad_attn_out);
    Tensor grad_input_from_ln1 = ln1.backward(grad_normalized1);

    return grad_input_direct + grad_input_from_ln1;
}

void TransformerBlock::update(float lr, int batch_size)
{
    mha.update(lr, batch_size);
    mlp.update(lr, batch_size);
    ln1.update(lr, batch_size);
    ln2.update(lr, batch_size);
}

void TransformerBlock::zero_grad()
{
    mha.zero_grad();
    mlp.zero_grad();
    ln1.zero_grad();
    ln2.zero_grad();
}