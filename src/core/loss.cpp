#include "../../include/core/loss.hpp"
#include "../../include/core/ops.hpp" // Para softmax
#include <cmath>
#include <vector>

float CrossEntropyLoss::forward(const Tensor &logits, int true_label_idx)
{
    int num_classes = logits.get_shape().back();

    // 1. Aplicar Softmax para obtener probabilidades
    softmax_output_cache_ = softmax(logits);

    // 2. Crear una representación one-hot de la etiqueta verdadera
    labels_one_hot_cache_ = Tensor({1, num_classes});
    labels_one_hot_cache_.zero_grad(); // Aunque no es un parámetro, lo usamos para almacenar el vector
    labels_one_hot_cache_.get_data()[true_label_idx] = 1.0f;

    // 3. Calcular la pérdida de entropía cruzada: -sum(y_true * log(y_pred))
    float loss = -std::log(softmax_output_cache_.get_data()[true_label_idx] + 1e-9); // Se suma epsilon para estabilidad numérica

    return loss;
}

Tensor CrossEntropyLoss::backward()
{
    // El gradiente de la entropía cruzada con softmax es sorprendentemente simple: (y_pred - y_true)
    return softmax_output_cache_ - labels_one_hot_cache_;
}