#pragma once
#include "../core/tensor.hpp"
#include <vector>

class FeedForwardNetwork
{
public:
    Tensor w1_;
    Tensor b1_;
    Tensor w2_;
    Tensor b2_;

private:
    // Cache para el backward pass
    Tensor input_cache_;
    Tensor hidden_activated_cache_;

public:
    FeedForwardNetwork(int d_model, int d_ff);
    Tensor forward(const Tensor &input);
    Tensor backward(const Tensor &grad_output);
    void get_parameters(std::vector<Tensor *> &params);
    void zero_all_grads();
};