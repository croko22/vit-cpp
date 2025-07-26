#include "../../include/model/mlp.h"
#include "../../include/core/activation.h"

MLP::MLP(int d_model, int hidden_dim)
    : fc1(d_model, hidden_dim),
      fc2(hidden_dim, d_model),
      ln(d_model),
      training(true)
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

    auto shape = grad_fc2.get_shape();
    Tensor grad_gelu_input(shape);

    auto &grad_gelu_data = grad_gelu_input.get_data();
    const auto &grad_fc2_data = grad_fc2.get_data();
    const auto &last_hidden_data = last_hidden.get_data();

    for (int i = 0; i < grad_fc2.get_size(); i++)
    {
        float x = last_hidden_data[i];
        float gelu_grad = Activation::gelu_derivative(x);
        grad_gelu_data[i] = grad_fc2_data[i] * gelu_grad;
    }

    return fc1.backward(grad_gelu_input);
}

void MLP::update(float lr, int batch_size)
{
    fc1.update(lr, batch_size);
    fc2.update(lr, batch_size);
    ln.update(lr, batch_size);
}

void MLP::zero_grad()
{
    fc1.zero_grad();
    fc2.zero_grad();
    ln.zero_grad();
}