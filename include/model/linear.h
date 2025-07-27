#ifndef LINEAR_H
#define LINEAR_H

#include "../../include/core/tensor.h"
#include "../../include/core/optimizer.h"
#include <algorithm>

class Linear
{
public:
    Tensor weight, bias;
    Tensor weight_grad, bias_grad;
    Tensor last_input;
    bool training;

    Linear(int in_features, int out_features);
    Tensor forward(const Tensor &input);
    Tensor backward(const Tensor &grad_output);

    void zero_grad();
    std::vector<Parameter> get_parameters();
};

#endif