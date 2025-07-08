#include "../../include/core/loss.hpp"
#include "../../include/core/ops.hpp" // Para softmax
#include <cmath>
#include <vector>

float CrossEntropyLoss::forward(const Tensor &logits, int true_label_idx)
{
    softmax_output_cache_ = softmax(logits); // ← asegúrate que retorna Tensor

    float *softmax_ptr = softmax_output_cache_.get_data();
    float correct_prob = softmax_ptr[true_label_idx]; // fila 0 asumida

    return -std::log(correct_prob + 1e-9f); // evitar log(0)
}

Tensor CrossEntropyLoss::backward(const Tensor &logits, int label)
{
    Tensor grad(logits.get_shape());
    grad.zero_data();

    int num_classes = logits.get_shape()[1];
    float *grad_ptr = grad.get_data();
    float *softmax_ptr = softmax_output_cache_.get_data();

    for (int i = 0; i < num_classes; ++i)
        grad_ptr[i] = softmax_ptr[i]; // fila 0 asumida

    grad_ptr[label] -= 1.0f;

    return grad;
}
