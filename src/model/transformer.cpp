#include "../../include/model/transformer.hpp"
#include "../../include/core/ops.hpp"

Transformer::Transformer(int num_layers, int d_model, int num_heads, int d_ff, int target_vocab_size)
    : num_layers_(num_layers),
      d_model_(d_model),
      final_linear_layer_weights_({d_model, target_vocab_size})
{
    for (int i = 0; i < num_layers; ++i)
    {
        encoder_layers_.emplace_back(d_model, num_heads, d_ff);
        decoder_layers_.emplace_back(d_model, num_heads, d_ff);
    }
}

Tensor Transformer::forward(const Tensor &src_tensor, const Tensor &tgt_tensor)
{
    Tensor encoder_output = encode(src_tensor);
    Tensor decoder_output = decode(tgt_tensor, encoder_output);
    Tensor logits = matmul(decoder_output, final_linear_layer_weights_);
    return logits;
}

Tensor Transformer::encode(const Tensor &src_tensor)
{
    // Crear una copia temporal usando el constructor de copia explícito
    // O mejor, trabajar directamente con referencias
    Tensor x({src_tensor.get_shape()});
    x.from_vector(src_tensor.to_vector());

    for (auto &layer : encoder_layers_)
    {
        x = layer.forward(x); // Usar move assignment
    }
    return x;
}

Tensor Transformer::decode(const Tensor &tgt_tensor, const Tensor &encoder_output)
{
    Tensor look_ahead_mask = create_look_ahead_mask(tgt_tensor.get_shape()[0]);

    // Crear copia temporal del target tensor
    Tensor y({tgt_tensor.get_shape()});
    y.from_vector(tgt_tensor.to_vector());

    for (auto &layer : decoder_layers_)
    {
        y = layer.forward(y, encoder_output, &look_ahead_mask, nullptr);
    }
    return y;
}

Tensor Transformer::create_look_ahead_mask(int size)
{
    Tensor mask({size, size});
    float *data = mask.get_data();
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            if (j > i)
            {
                data[i * size + j] = -1e9f;
            }
            else
            {
                data[i * size + j] = 0.0f;
            }
        }
    }
    return mask;
}