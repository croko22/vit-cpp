#ifndef MULTIHEAD_ATTENTION_H
#define MULTIHEAD_ATTENTION_H

#include "../../include/core/tensor.h"
#include "linear.h"
#include <vector>

class MultiHeadAttention
{
public:
    int d_model;
    int num_heads;
    int d_k;

    Linear q_proj, k_proj, v_proj;
    Linear out_proj;

    Tensor last_q, last_k, last_v;
    Tensor last_attention_weights;

    MultiHeadAttention(int d_model, int num_heads);
    Tensor forward(const Tensor &x);
    Tensor backward(const Tensor &grad_output);

    void zero_grad();

    std::vector<Parameter> get_parameters();
};

#endif