#pragma once

#include "../core/optimizer.h"

class Adam : public Optimizer
{
private:
    float beta1, beta2, epsilon;
    int t;
    std::vector<Tensor> m, v;

public:
    Adam(std::vector<Parameter> params, float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f);
    void step() override;
};