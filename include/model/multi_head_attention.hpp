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
public:
    Tensor w_q_; // Proyección para Query
    Tensor w_k_; // Proyección para Key
    Tensor w_v_; // Proyección para Value
    Tensor w_o_; // Proyección de salida

private:
    int d_model_;
    int num_heads_;
    float scale_;

    // Cache para el backward pass
    Tensor query_input_;       // Entrada original Q
    Tensor key_input_;         // Entrada original K
    Tensor value_input_;       // Entrada original V
    Tensor q_proj_;            // Q después de la proyección lineal
    Tensor k_proj_;            // K después de la proyección lineal
    Tensor v_proj_;            // V después de la proyección lineal
    Tensor attention_weights_; // Pesos de atención (salida del softmax)
    Tensor context_;           // Salida de la atención (attention_weights @ V)

public:
    MultiHeadAttention(int d_model, int num_heads);
    Tensor forward(const Tensor &query, const Tensor &key, const Tensor &value, const Tensor *mask = nullptr);
    std::tuple<Tensor, Tensor, Tensor> backward(const Tensor &grad_output);
    void get_parameters(std::vector<Tensor *> &params);
    void zero_all_grads();
};

#endif // MULTI_HEAD_ATTENTION_HPP