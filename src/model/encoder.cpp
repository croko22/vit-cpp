#include "../../include/model/encoder.h" // Asegúrate que la ruta sea correcta

// --- Constructor Corregido ---
// Ahora acepta todos los parámetros necesarios y los pasa a las capas internas.
TransformerBlock::TransformerBlock(int d_model, int num_heads, int d_ff)
    : mha(d_model, num_heads), // Inicializa la capa MultiHeadAttention
      mlp(d_model, d_ff),      // Inicializa el MLP
      ln1(d_model),            // Inicializa la primera LayerNorm
      ln2(d_model)             // Inicializa la segunda LayerNorm
{
    // El cuerpo del constructor puede estar vacío si todo se inicializa arriba
}

// --- Forward Pass con la Arquitectura "Pre-LN" ---
// Esta es una arquitectura moderna y estable: y = x + SubLayer(LayerNorm(x))
Tensor TransformerBlock::forward(const Tensor &input)
{
    // Guarda el input para la conexión residual
    last_input = input;

    // --- Primer Bloque: Multi-Head Attention ---
    // 1. Normaliza la entrada
    last_normalized1 = ln1.forward(input);
    // 2. Pasa por la capa de atención
    last_attn_out = mha.forward(last_normalized1);
    // 3. Añade la conexión residual
    last_residual1 = input + last_attn_out;

    // --- Segundo Bloque: MLP (Feed-Forward) ---
    // 1. Normaliza la salida del bloque anterior
    last_normalized2 = ln2.forward(last_residual1);
    // 2. Pasa por el MLP
    Tensor mlp_out = mlp.forward(last_normalized2);
    // 3. Añade la segunda conexión residual
    return last_residual1 + mlp_out;
}

// --- Backward Pass ---
// Sigue la regla de la cadena en orden inverso al forward pass
Tensor TransformerBlock::backward(const Tensor &grad_output)
{
    // --- Backprop a través del segundo bloque (MLP) ---
    // El gradiente se bifurca por la conexión residual:
    Tensor grad_residual1_from_mlp = grad_output; // Ruta directa (skip connection)
    Tensor grad_mlp_out = grad_output;            // Ruta a través del MLP

    // Propaga por las capas
    Tensor grad_normalized2 = mlp.backward(grad_mlp_out);
    Tensor grad_residual1_from_ln2 = ln2.backward(grad_normalized2);

    // Suma los gradientes de ambas rutas. ¡ESTE PUNTO ES CRÍTICO!
    Tensor grad_residual1 = grad_residual1_from_mlp + grad_residual1_from_ln2;

    // --- Backprop a través del primer bloque (MHA) ---
    // El gradiente se vuelve a bifurcar
    Tensor grad_input_direct = grad_residual1; // Ruta directa
    Tensor grad_attn_out = grad_residual1;     // Ruta a través del MHA

    Tensor grad_normalized1 = mha.backward(grad_attn_out);
    Tensor grad_input_from_ln1 = ln1.backward(grad_normalized1);

    // Suma los gradientes finales. ¡ESTE PUNTO TAMBIÉN ES CRÍTICO!
    return grad_input_direct + grad_input_from_ln1;
}

// --- Métodos de Actualización y reseteo ---
void TransformerBlock::update(float lr, int batch_size)
{
    mha.update(lr, batch_size);
    mlp.update(lr, batch_size);
    ln1.update(lr, batch_size);
    ln2.update(lr, batch_size);
}

void TransformerBlock::zero_grad()
{
    mha.zero_grad();
    mlp.zero_grad();
    ln1.zero_grad();
    ln2.zero_grad();
}