#pragma once
#include "../core/tensor.hpp"
#include <vector>

class FeedForwardNetwork
{
private:
    Tensor w1_;
    Tensor b1_;
    Tensor w2_;
    Tensor b2_;

public:
    FeedForwardNetwork(int d_model, int d_ff);
    Tensor forward(const Tensor &input);
};