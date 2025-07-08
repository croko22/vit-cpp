#pragma once

#include "tensor.hpp"

class CrossEntropyLoss
{
private:
    Tensor softmax_output_cache_;
    Tensor labels_one_hot_cache_;

public:
    CrossEntropyLoss() = default;

    /**
     * @brief Calcula la pérdida de entropía cruzada.
     * @param logits La salida cruda del modelo (sin softmax).
     * @param true_label_idx El índice de la clase correcta (ej. 7 para el dígito '7').
     * @return El valor de la pérdida (un escalar).
     */
    float forward(const Tensor &logits, int true_label_idx);

    /**
     * @brief Calcula el gradiente inicial (dLoss / dLogits).
     * @return Un tensor con el gradiente para iniciar el backward pass.
     */
    Tensor backward(const Tensor &logits, int label);
};