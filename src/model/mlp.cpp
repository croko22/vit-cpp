#include "../../include/model/mlp.h"
#include "../../include/core/activation.h"
#include "../../include/core/random.h"

MLP::MLP(int d_model, int hidden_dim, float dropout_probability)
    : fc1(d_model, hidden_dim),
      fc2(hidden_dim, d_model),
      ln(d_model),
      training(true),
      drop_prob(dropout_probability)
{
}

Tensor MLP::forward(const Tensor &input)
{
    last_hidden = fc1.forward(input);
    last_activated = Activation::apply(last_hidden, Activation::gelu);

    Tensor after_dropout(last_activated.get_shape());
    if (training && drop_prob > 0.0f)
    {
        dropout_mask = Tensor(last_activated.get_shape());
        float scale = 1.0f / (1.0f - drop_prob);

        for (int i = 0; i < last_activated.get_size(); ++i)
        {

            if (Random::uniform(0.0f, 1.0f) > drop_prob)
            {
                dropout_mask.data[i] = scale;
                after_dropout.data[i] = last_activated.data[i] * scale;
            }
            else
            {
                dropout_mask.data[i] = 0.0f;
                after_dropout.data[i] = 0.0f;
            }
        }
    }
    else
    {
        after_dropout = last_activated;
    }

    Tensor output = fc2.forward(after_dropout);
    return ln.forward(output);
}

Tensor MLP::backward(const Tensor &grad_output)
{
    Tensor grad_ln = ln.backward(grad_output);
    Tensor grad_fc2 = fc2.backward(grad_ln);

    Tensor grad_after_dropout(grad_fc2.get_shape());
    if (training && drop_prob > 0.0f)
    {
        for (int i = 0; i < grad_fc2.get_size(); ++i)
        {
            grad_after_dropout.data[i] = grad_fc2.data[i] * dropout_mask.data[i];
        }
    }
    else
    {
        grad_after_dropout = grad_fc2;
    }

    Tensor grad_gelu_input(grad_after_dropout.get_shape());

    const auto &grad_from_dropout = grad_after_dropout.get_data();
    const auto &last_hidden_data = last_hidden.get_data();
    auto &grad_gelu_data = grad_gelu_input.get_data();

    for (int i = 0; i < grad_after_dropout.get_size(); i++)
    {
        float x = last_hidden_data[i];
        float gelu_grad_val = Activation::gelu_derivative(x);
        grad_gelu_data[i] = grad_from_dropout[i] * gelu_grad_val;
    }

    return fc1.backward(grad_gelu_input);
}

std::vector<Parameter> MLP::get_parameters()
{
    auto params = fc1.get_parameters();
    auto fc2_params = fc2.get_parameters();
    auto ln_params = ln.get_parameters();

    params.insert(params.end(), fc2_params.begin(), fc2_params.end());
    params.insert(params.end(), ln_params.begin(), ln_params.end());

    return params;
}

void MLP::zero_grad()
{
    fc1.zero_grad();
    fc2.zero_grad();
    ln.zero_grad();
}