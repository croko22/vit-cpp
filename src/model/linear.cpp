#include "../../include/model/linear.h"

Linear::Linear(int in_features, int out_features) : weight(out_features, in_features),
                                                    bias(out_features, 1),
                                                    weight_grad(out_features, in_features),
                                                    bias_grad(out_features, 1),
                                                    training(true)
{
    weight.xavier_init();
    bias.zero();
}

Tensor Linear::forward(const Tensor &input)
{
    if (training)
    {
        last_input = input;
    }
    Tensor result = weight * input.transpose();
    for (int i = 0; i < result.rows; i++)
    {
        for (int j = 0; j < result.cols; j++)
        {
            result(i, j) += bias(i, 0);
        }
    }
    return result.transpose();
}

Tensor Linear::backward(const Tensor &grad_output)
{
    Tensor grad_w = grad_output.transpose() * last_input;
    weight_grad = weight_grad + grad_w;

    for (int i = 0; i < grad_output.cols; i++)
    {
        for (int j = 0; j < grad_output.rows; j++)
        {
            bias_grad(i, 0) += grad_output(j, i);
        }
    }
    return grad_output * weight;
}

void Linear::update(float lr)
{
    // TO CHANGUE: Borrar y volver al anterior en caso de falla
    float max_norm = 1.0f;  // Ajusta este valor según necesidad
    float grad_norm = sqrt(weight_grad.norm() + bias_grad.norm());

    if (grad_norm > max_norm) {
        float scale = max_norm / grad_norm;
        // Escala todos los componentes del gradiente
        for (int i = 0; i < weight_grad.rows * weight_grad.cols; i++) {
            weight_grad.data[i] *= scale;
        }
        for (int i = 0; i < bias_grad.rows; i++) {
            bias_grad(i, 0) *= scale;
        }
    }

    // 2. Actualización estándar
    for (int i = 0; i < weight.rows * weight.cols; i++) {
        weight.data[i] -= lr * weight_grad.data[i];
    }
    for (int i = 0; i < bias.rows; i++) {
        bias(i, 0) -= lr * bias_grad(i, 0);
    }
}

void Linear::zero_grad()
{
    weight_grad.zero();
    bias_grad.zero();
}
