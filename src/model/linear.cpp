#include "../../include/model/linear.h"

Linear::Linear(int in_features, int out_features) : weight(out_features, in_features),
                                                    bias(out_features, 1),
                                                    weight_grad(out_features, in_features),
                                                    bias_grad(out_features, 1),
                                                    training(true)
{
    weight.xavier_init();
    bias.zero();
}

Tensor Linear::forward(const Tensor &input)
{
    if (training)
    {
        last_input = input;
    }
    Tensor result = weight * input.transpose();
    for (int i = 0; i < result.rows; i++)
    {
        for (int j = 0; j < result.cols; j++)
        {
            result(i, j) += bias(i, 0);
        }
    }
    return result.transpose();
}

Tensor Linear::backward(const Tensor &grad_output)
{
    Tensor grad_w = grad_output.transpose() * last_input;
    weight_grad = weight_grad + grad_w;

    for (int i = 0; i < grad_output.cols; i++)
    {
        for (int j = 0; j < grad_output.rows; j++)
        {
            bias_grad(i, 0) += grad_output(j, i);
        }
    }
    return grad_output * weight;
}

void Linear::update(float lr)
{
    float max_grad = 1.0f;
    for (int i = 0; i < weight.rows * weight.cols; i++)
    {
        weight_grad.data[i] = std::max(-max_grad, std::min(max_grad, weight_grad.data[i]));
        weight.data[i] -= lr * weight_grad.data[i];
    }
    for (int i = 0; i < bias.rows; i++)
    {
        bias_grad(i, 0) = std::max(-max_grad, std::min(max_grad, bias_grad(i, 0)));
        bias(i, 0) -= lr * bias_grad(i, 0);
    }
}

void Linear::zero_grad()
{
    weight_grad.zero();
    bias_grad.zero();
}
