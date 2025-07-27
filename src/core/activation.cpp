#include "../../include/core/activation.h"
#include "../../include/core/tensor.h"
#include "../../include/core/random.h"
#include <cmath>
#include <algorithm>
#include <vector>

float Activation::relu(float x)
{
    return std::max(0.0f, x);
}

float Activation::relu_derivative(float x)
{
    return x > 0 ? 1.0f : 0.0f;
}

float Activation::gelu(float x)
{
    return 0.5f * x * (1.0f + tanh(sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
}

float Activation::gelu_derivative(float x)
{
    float tanh_arg = sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x);
    float tanh_val = tanh(tanh_arg);
    float sech_sq = 1.0f - tanh_val * tanh_val;
    return 0.5f * (1.0f + tanh_val) + 0.5f * x * sech_sq * sqrt(2.0f / M_PI) * (1.0f + 0.134145f * x * x);
}

// Función genérica para aplicar una operación elemento a elemento
Tensor Activation::apply(const Tensor &input, float (*func)(float))
{
    // 1. Crea un tensor resultado con la misma forma que el de entrada
    Tensor result(input.get_shape());

    // 2. Accede a los datos a través de los getters
    auto &result_data = result.get_data();
    const auto &input_data = input.get_data();

    // 3. Itera sobre todos los elementos usando get_size()
    for (int i = 0; i < input.get_size(); i++)
    {
        result_data[i] = func(input_data[i]);
    }
    return result;
}

// Softmax que funciona para tensores N-D (opera sobre la última dimensión)
Tensor Activation::softmax(const Tensor &input)
{
    auto shape = input.get_shape();
    if (shape.empty())
    {
        return Tensor(); // Devuelve tensor vacío si la entrada está vacía
    }

    Tensor result(shape);
    auto &result_data = result.get_data();
    const auto &input_data = input.get_data();

    int last_dim = shape.back();
    int outer_size = input.get_size() / last_dim;

    // Itera sobre cada "fila" o "slice" a lo largo de las dimensiones exteriores
    for (int i = 0; i < outer_size; ++i)
    {
        int offset = i * last_dim;

        // 1. Encontrar el valor máximo en el slice actual para estabilidad numérica
        float max_val = input_data[offset];
        for (int j = 1; j < last_dim; ++j)
        {
            max_val = std::max(max_val, input_data[offset + j]);
        }

        // 2. Calcular exp y la suma
        float sum = 0.0f;
        for (int j = 0; j < last_dim; ++j)
        {
            float val = std::exp(input_data[offset + j] - max_val);
            result_data[offset + j] = val;
            sum += val;
        }

        // 3. Normalizar
        for (int j = 0; j < last_dim; ++j)
        {
            result_data[offset + j] /= sum;
        }
    }
    return result;
}

// Gradiente de Softmax que funciona para tensores N-D
Tensor Activation::softmax_grad(const Tensor &softmax_output, const Tensor &grad_output)
{
    if (softmax_output.get_shape() != grad_output.get_shape())
    {
        throw std::invalid_argument("softmax_grad: las formas deben coincidir.");
    }

    auto shape = softmax_output.get_shape();
    Tensor result(shape);

    auto &result_data = result.get_data();
    const auto &s_data = softmax_output.get_data();
    const auto &g_data = grad_output.get_data();

    int last_dim = shape.back();
    int outer_size = softmax_output.get_size() / last_dim;

    // Itera sobre cada slice, igual que en la función softmax
    for (int i = 0; i < outer_size; ++i)
    {
        int offset = i * last_dim;

        // Calcular el término de la suma (dot product del slice)
        float sum_term = 0.0f;
        for (int k = 0; k < last_dim; ++k)
        {
            sum_term += s_data[offset + k] * g_data[offset + k];
        }

        // Calcular el gradiente para cada elemento en el slice
        for (int j = 0; j < last_dim; ++j)
        {
            float s_ij = s_data[offset + j];
            float grad_ij = g_data[offset + j];
            result_data[offset + j] = s_ij * (grad_ij - sum_term);
        }
    }

    return result;
}

Tensor Activation::dropout(const Tensor &input, float drop_prob, bool training)
{
    if (!training || drop_prob == 0.0f)
    {
        return input; // No hacer nada durante la inferencia o si la probabilidad es 0
    }

    Tensor result(input.get_shape());
    float scale = 1.0f / (1.0f - drop_prob); // Inverted Dropout

    for (int i = 0; i < input.get_size(); ++i)
    {
        if (Random::uniform(0.0f, 1.0f) > drop_prob)
        {
            result.data[i] = input.data[i] * scale;
        }
        else
        {
            result.data[i] = 0.0f;
        }
    }
    return result;
}