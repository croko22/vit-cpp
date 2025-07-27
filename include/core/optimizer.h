#pragma once

#include "tensor.h"
#include <vector>

struct Parameter
{
    Tensor *weight;
    Tensor *grad;
};

class Optimizer
{
protected:
    std::vector<Parameter> params;
    float lr;

public:
    Optimizer(std::vector<Parameter> params, float learning_rate)
        : params(params), lr(learning_rate) {}

    virtual ~Optimizer() = default;

    virtual void step() = 0;

    void zero_grad()
    {
        for (auto &p : params)
        {
            p.grad->zero();
        }
    }
};