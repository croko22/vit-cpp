#ifndef OPS_HPP
#define OPS_HPP

#include "tensor.hpp"

/**
 * @brief Realiza la multiplicación de matrices (MatMul).
 * Soporta (secuencia, features) @ (features, features) -> (secuencia, features)
 * @param a Tensor de entrada A.
 * @param b Tensor de entrada B.
 * @return Un nuevo Tensor con el resultado.
 */
Tensor matmul(const Tensor &a, const Tensor &b);
std::pair<Tensor, Tensor> matmul_backward(const Tensor &grad_output, const Tensor &a, const Tensor &b);

/**
 * @brief Aplica la función Softmax a la última dimensión del tensor.
 * @param input El tensor de entrada.
 * @return Un nuevo Tensor con probabilidades.
 */
Tensor softmax(const Tensor &input);
Tensor softmax_backward(const Tensor &grad_output, const Tensor &softmax_output);

/**
 * @brief Aplica la función de activación ReLU elemento por elemento.
 * @param input El tensor de entrada.
 * @return Un nuevo Tensor con ReLU aplicado.
 */
Tensor relu(const Tensor &input);
Tensor relu_backward(const Tensor &grad_output, const Tensor &relu_output);

/**
 * @brief Suma a lo largo de un eje específico.
 * @param input El tensor de entrada.
 * @param axis El eje a lo largo del cual sumar.
 * @param keep_dims Si se deben mantener las dimensiones originales.
 * @return Un nuevo Tensor con la suma realizada.
 */
Tensor sum(const Tensor &input, int axis, bool keep_dims = false);

#endif // OPS_HPP
