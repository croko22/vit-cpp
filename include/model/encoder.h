#ifndef TRANSFORMER_BLOCK_H
#define TRANSFORMER_BLOCK_H

#include "../../include/core/tensor.h"
#include "multihead_attention.h" // Capa de Multi-Head Attention
#include "mlp.h"                 // Red Feed-Forward
#include "layernorm.h"

class TransformerBlock
{
public:
    // --- Interfaz Pública ---
    // El constructor ahora toma todos los parámetros necesarios.
    TransformerBlock(int d_model, int num_heads, int d_ff);

    Tensor forward(const Tensor &input);
    Tensor backward(const Tensor &grad_output);
    void update(float lr, int batch_size = 1);
    void zero_grad();

    // --- Miembros Privados (Encapsulamiento) ---
    // Capas del bloque
    MultiHeadAttention mha; // Corregido: El nombre de la clase es MultiHeadAttention
    MLP mlp;
    LayerNorm ln1, ln2;

    // Tensores cacheados para el backward pass
    Tensor last_input;
    Tensor last_attn_out;
    Tensor last_residual1;
    Tensor last_normalized1;
    Tensor last_normalized2;
};

#endif // TRANSFORMER_BLOCK_H