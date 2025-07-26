#ifndef MULTIHEAD_ATTENTION_H
#define MULTIHEAD_ATTENTION_H

#include "../../include/core/tensor.h"
#include "linear.h"
#include <vector>

class MultiHeadAttention
{
public:
    int d_model;   // Dimensión del modelo
    int num_heads; // Número de cabezas
    int d_k;       // Dimensión por cabeza (d_model / num_heads)

    Linear q_proj, k_proj, v_proj; // Proyecciones Q, K, V
    Linear out_proj;               // Proyección final

    Tensor last_q, last_k, last_v;
    Tensor last_attention_weights;

    MultiHeadAttention(int d_model, int num_heads);
    Tensor forward(const Tensor &x);
    Tensor backward(const Tensor &grad_output);
    void update(float lr, int batch_size = 1);
    void zero_grad();
};

#endif // MULTIHEAD_ATTENTION_H