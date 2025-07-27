#pragma once

#include "../core/optimizer.h"

class SGD : public Optimizer
{
public:
    SGD(std::vector<Parameter> params, float lr = 0.01f);
    void step() override;
};