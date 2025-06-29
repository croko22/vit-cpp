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

/**
 * @brief Aplica la función Softmax a la última dimensión del tensor.
 * @param input El tensor de entrada.
 * @return Un nuevo Tensor con probabilidades.
 */
Tensor softmax(const Tensor &input);

/**
 * @brief Aplica la función de activación ReLU elemento por elemento.
 * @param input El tensor de entrada.
 * @return Un nuevo Tensor con ReLU aplicado.
 */
Tensor relu(const Tensor &input);

#endif // OPS_HPP