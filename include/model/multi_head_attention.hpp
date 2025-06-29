#ifndef MULTI_HEAD_ATTENTION_HPP
#define MULTI_HEAD_ATTENTION_HPP

#include "../core/tensor.hpp"

/**
 * @class MultiHeadAttention
 * @brief Implementa el mecanismo de Multi-Head Self-Attention.
 * NOTA: Para simplificar, esta versión inicial NO divide en cabezales.
 * Implementa un Scaled Dot-Product Attention simple. La división en
 * cabezales es el siguiente paso lógico.
 */
class MultiHeadAttention
{
private:
    int d_model_;
    int num_heads_;
    float scale_;

    Tensor w_q_; // Proyección para Query
    Tensor w_k_; // Proyección para Key
    Tensor w_v_; // Proyección para Value
    Tensor w_o_; // Proyección de salida

public:
    MultiHeadAttention(int d_model, int num_heads);
    Tensor forward(const Tensor &input);
};

#endif // MULTI_HEAD_ATTENTION_HPP