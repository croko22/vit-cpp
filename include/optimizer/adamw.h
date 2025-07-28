#pragma once

#include "../core/optimizer.h"
#include <vector>

class AdamW : public Optimizer
{
private:
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;

    int t;

    std::vector<Tensor> m;
    std::vector<Tensor> v;

public:
    AdamW(std::vector<Parameter> params, float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f, float wd = 0.01f);
    void step() override;
};