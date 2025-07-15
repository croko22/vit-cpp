#include "../../include/model/mlp.h"
#include "../../include/core/activation.h"
#include <iostream>

MLP::MLP(int d_model, int hidden_dim)
    : fc1(d_model, hidden_dim), fc2(hidden_dim, d_model),
      ln(d_model), training(true)
{
}

Tensor MLP::forward(const Tensor &input)
{
    last_hidden = fc1.forward(input);
    last_activated = Activation::apply(last_hidden, Activation::gelu);
    Tensor output = fc2.forward(last_activated);
    return ln.forward(output);
}

Tensor MLP::backward(const Tensor &grad_output)
{
    Tensor grad_ln = ln.backward(grad_output);
    Tensor grad_fc2 = fc2.backward(grad_ln);

    Tensor grad_gelu_input(grad_fc2.rows, grad_fc2.cols);
    for (int i = 0; i < grad_fc2.rows; i++)
    {
        for (int j = 0; j < grad_fc2.cols; j++)
        {
            float x = last_hidden(i, j);
            float gelu_grad = Activation::gelu_derivative(x);
            grad_gelu_input(i, j) = grad_fc2(i, j) * gelu_grad;
        }
    }
    return fc1.backward(grad_gelu_input);
}

void MLP::update(float lr)
{
    fc1.update(lr);
    fc2.update(lr);
    ln.update(lr);
}

void MLP::zero_grad()
{
    fc1.zero_grad();
    fc2.zero_grad();
    ln.zero_grad();
}
