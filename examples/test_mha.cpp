#include <iostream>
#include <vector>
#include <algorithm>

#include "../include/model/multihead_attention.h"
#include "../include/core/tensor.h"

int main()
{
    int d_model = 64;
    int seq_len = 2;
    int num_heads = 8;

    MultiHeadAttention mha(d_model, num_heads);

    Tensor x({seq_len, d_model});

    x(0, 0) = 0.1;
    x(0, 1) = 0.2;
    x(0, 2) = 0.3;
    x(0, 3) = 0.4;
    x(1, 0) = 0.5;
    x(1, 1) = 0.6;
    x(1, 2) = 0.7;
    x(1, 3) = 0.8;

    std::cout << "Input X:" << std::endl;
    x.print();

    Tensor out = mha.forward(x);
    std::cout << "\nForward output:" << std::endl;
    out.print();

    Tensor grad_out(out.get_shape());

    auto &grad_data = grad_out.get_data();
    std::fill(grad_data.begin(), grad_data.end(), 1.0f);

    Tensor grad_in = mha.backward(grad_out);
    std::cout << "\nBackward grad_input:" << std::endl;
    grad_in.print();

    mha.update(0.01);

    std::cout << "\nUpdate step done." << std::endl;

    return 0;
}