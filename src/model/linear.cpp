#include "../../include/model/linear.h"
#include <vector>
#include <numeric>

Linear::Linear(int in_features, int out_features) : weight({out_features, in_features}),
                                                    bias({out_features}),
                                                    weight_grad({out_features, in_features}),
                                                    bias_grad({out_features}),
                                                    training(true)
{

    Tensor::xavier_init(this->weight);
    this->bias.zero();
}

Tensor Linear::forward(const Tensor &input)
{
    if (training)
    {
        last_input = input;
    }

    auto input_shape = input.get_shape();

    std::vector<int> original_shape(input_shape.begin(), input_shape.end() - 1);

    int batch_size = 1;
    for (size_t i = 0; i < original_shape.size(); ++i)
    {
        batch_size *= original_shape[i];
    }

    int in_features = input_shape.back();
    Tensor input_2d = input.reshape({batch_size, in_features});

    Tensor result_2d = input_2d * weight.transpose(0, 1);

    auto &result_data = result_2d.get_data();
    const auto &bias_data = bias.get_data();
    int out_features = weight.get_shape()[0];
    for (int i = 0; i < batch_size; ++i)
    {
        for (int j = 0; j < out_features; ++j)
        {
            result_data[i * out_features + j] += bias_data[j];
        }
    }

    original_shape.push_back(out_features);
    return result_2d.reshape(original_shape);
}

Tensor Linear::backward(const Tensor &grad_output)
{
    auto output_shape = grad_output.get_shape();
    auto input_shape = last_input.get_shape();

    int batch_size = 1;
    for (size_t i = 0; i < output_shape.size() - 1; ++i)
        batch_size *= output_shape[i];

    int out_features = output_shape.back();
    int in_features = input_shape.back();

    Tensor grad_output_2d = grad_output.reshape({batch_size, out_features});
    Tensor last_input_2d = last_input.reshape({batch_size, in_features});

    Tensor grad_input_2d = grad_output_2d * weight;

    Tensor grad_w = grad_output_2d.transpose(0, 1) * last_input_2d;
    this->weight_grad = this->weight_grad + grad_w;

    auto &bias_grad_data = bias_grad.get_data();
    const auto &grad_output_data = grad_output_2d.get_data();
    for (int j = 0; j < out_features; ++j)
    {
        float sum = 0.0f;
        for (int i = 0; i < batch_size; ++i)
        {
            sum += grad_output_data[i * out_features + j];
        }
        bias_grad_data[j] += sum;
    }

    return grad_input_2d.reshape(input_shape);
}

std::vector<Parameter> Linear::get_parameters()
{
    return {
        {&weight, &weight_grad},
        {&bias, &bias_grad}};
}

void Linear::zero_grad()
{
    weight_grad.zero();
    bias_grad.zero();
}