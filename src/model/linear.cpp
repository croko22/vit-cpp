#include "../../include/model/linear.h" // Ajusta tu ruta
#include <vector>
#include <numeric>

Linear::Linear(int in_features, int out_features) : // Usa el constructor de Tensor con std::vector<int> para la forma
                                                    weight({out_features, in_features}),
                                                    bias({out_features}), // El bias es 1D, de tamaño out_features
                                                    weight_grad({out_features, in_features}),
                                                    bias_grad({out_features}),
                                                    training(true)
{
    // Usa la función estática de inicialización
    Tensor::xavier_init(this->weight);
    this->bias.zero(); // El bias se inicializa a cero
}

Tensor Linear::forward(const Tensor &input)
{
    if (training)
    {
        last_input = input;
    }

    auto input_shape = input.get_shape();
    // Guarda la forma original de entrada (sin la última dimensión)
    std::vector<int> original_shape(input_shape.begin(), input_shape.end() - 1);

    // Calcula el tamaño del batch (producto de las dims excepto la última)
    int batch_size = 1;
    for (size_t i = 0; i < original_shape.size(); ++i)
    {
        batch_size *= original_shape[i];
    }

    // Aplana la entrada a 2D: [N, in_features]
    int in_features = input_shape.back();
    Tensor input_2d = input.reshape({batch_size, in_features});

    // Multiplicación de matrices: [N, in_features] * [out_features, in_features]^T -> [N, out_features]
    Tensor result_2d = input_2d * weight.transpose(0, 1);

    // Broadcasting del bias
    // Suma el vector de bias a cada fila del resultado
    auto &result_data = result_2d.get_data();
    const auto &bias_data = bias.get_data();
    int out_features = weight.get_shape()[0];
    for (int i = 0; i < batch_size; ++i)
    {
        for (int j = 0; j < out_features; ++j)
        {
            result_data[i * out_features + j] += bias_data[j];
        }
    }

    // Devuelve el resultado a su forma original N-D
    original_shape.push_back(out_features);
    return result_2d.reshape(original_shape);
}

Tensor Linear::backward(const Tensor &grad_output)
{
    auto output_shape = grad_output.get_shape();
    auto input_shape = last_input.get_shape();

    int batch_size = 1;
    for (size_t i = 0; i < output_shape.size() - 1; ++i)
        batch_size *= output_shape[i];

    int out_features = output_shape.back();
    int in_features = input_shape.back();

    // Aplana grad_output y last_input a 2D para los cálculos
    Tensor grad_output_2d = grad_output.reshape({batch_size, out_features});
    Tensor last_input_2d = last_input.reshape({batch_size, in_features});

    // 1. Gradiente del input: grad_output * W
    // [N, out_features] * [out_features, in_features] -> [N, in_features]
    Tensor grad_input_2d = grad_output_2d * weight;

    // 2. Gradiente de los pesos: grad_output^T * last_input
    // [out_features, N] * [N, in_features] -> [out_features, in_features]
    Tensor grad_w = grad_output_2d.transpose(0, 1) * last_input_2d;
    this->weight_grad = this->weight_grad + grad_w;

    // 3. Gradiente del bias: suma de grad_output a lo largo del batch
    auto &bias_grad_data = bias_grad.get_data();
    const auto &grad_output_data = grad_output_2d.get_data();
    for (int j = 0; j < out_features; ++j)
    {
        float sum = 0.0f;
        for (int i = 0; i < batch_size; ++i)
        {
            sum += grad_output_data[i * out_features + j];
        }
        bias_grad_data[j] += sum;
    }

    // Devuelve el gradiente del input con su forma original
    return grad_input_2d.reshape(input_shape);
}

void Linear::update(float lr, int batch_size)
{
    auto &w_data = weight.get_data();
    const auto &wg_data = weight_grad.get_data();
    auto &b_data = bias.get_data();
    const auto &bg_data = bias_grad.get_data();

    float scale = lr / batch_size;

    for (size_t i = 0; i < w_data.size(); ++i)
    {
        w_data[i] -= scale * wg_data[i];
    }
    for (size_t i = 0; i < b_data.size(); ++i)
    {
        b_data[i] -= scale * bg_data[i];
    }
}

void Linear::zero_grad()
{
    weight_grad.zero();
    bias_grad.zero();
}