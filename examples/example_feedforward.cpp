#include "../include/model/feedforward.hpp"
#include <iostream>
#include <vector>
#include <cassert> // Para las verificaciones (asserts)

// Función auxiliar para verificar las formas de los tensores
void check_shape(const Tensor &t, const std::vector<int> &expected, const std::string &name)
{
    const auto &actual = t.get_shape();
    assert(actual == expected);
    std::cout << "✓ Shape de " << name << " es correcta: [";
    for (size_t i = 0; i < actual.size(); ++i)
    {
        std::cout << actual[i] << (i == actual.size() - 1 ? "" : ", ");
    }
    std::cout << "]" << std::endl;
}

int main()
{
    std::cout << "=== FeedForward Network Full Test (Forward & Backward) ===" << std::endl;

    // --- 1. Configuración ---
    int d_model = 64;
    int d_ff = 128;
    int seq_len = 17; // Como en tus logs de error

    FeedForwardNetwork ffn(d_model, d_ff);

    // Inicializar los gradientes para la prueba
    ffn.w1_.init_grad();
    ffn.b1_.init_grad();
    ffn.w2_.init_grad();
    ffn.b2_.init_grad();

    // --- 2. Forward Pass ---
    std::cout << "\n--- Probando Forward Pass ---" << std::endl;
    Tensor input({seq_len, d_model});
    // (No necesitamos llenar el input con datos para esta prueba de formas)

    Tensor output = ffn.forward(input);
    output.print("Salida del Forward");
    check_shape(output, {seq_len, d_model}, "Output");

    // --- 3. Backward Pass ---
    std::cout << "\n--- Probando Backward Pass ---" << std::endl;

    // Crear un gradiente falso que simula venir de la capa siguiente.
    // Debe tener la misma forma que la salida del forward.
    Tensor grad_output({seq_len, d_model});
    grad_output.print("Gradiente de entrada (simulado)");

    // ¡Ejecutar el backward pass!
    Tensor grad_input = ffn.backward(grad_output);
    grad_input.print("Gradiente de salida (hacia la capa anterior)");

    // --- 4. Verificación de Formas de los Gradientes ---
    std::cout << "\n--- Verificando las formas de los gradientes calculados ---" << std::endl;

    // El gradiente de un parámetro debe tener la misma forma que el parámetro mismo.
    check_shape(*(ffn.w1_.grad_), ffn.w1_.get_shape(), "Gradiente de W1");
    check_shape(*(ffn.b1_.grad_), ffn.b1_.get_shape(), "Gradiente de B1");
    check_shape(*(ffn.w2_.grad_), ffn.w2_.get_shape(), "Gradiente de W2");
    check_shape(*(ffn.b2_.grad_), ffn.b2_.get_shape(), "Gradiente de B2");

    // El gradiente devuelto debe tener la misma forma que la entrada original.
    check_shape(grad_input, input.get_shape(), "Gradiente de Input");

    std::cout << "\n\n✓✓✓ ¡Prueba completa del FeedForwardNetwork finalizada con éxito! ✓✓✓" << std::endl;

    return 0;
}