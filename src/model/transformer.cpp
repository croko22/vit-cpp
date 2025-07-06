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
    }
}

Tensor Transformer::forward(const Tensor &src_tensor, const Tensor &tgt_tensor)
{
    Tensor encoder_output = encode(src_tensor);
    decoder_output = decode(tgt_tensor, encoder_output);
    Tensor logits = matmul(decoder_output, final_linear_layer_weights_);
    return logits;
}

Tensor Transformer::encode(const Tensor &src_tensor)
{
    Tensor x({src_tensor.get_shape()});
    x.from_vector(src_tensor.to_vector());

    for (auto &layer : encoder_layers_)
    {
        x = layer.forward(x);
    }
    return x;
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
void Transformer::backward(const Tensor &grad_output)
{
    // 1. Backward del Linear final
    Tensor grad_decoder_output = matmul(grad_output, final_linear_layer_weights_.transpose());

    // dL/dW = Output_decoder^T * dL/dOutput
    Tensor grad_final_weights = matmul(decoder_output.transpose(), grad_output);
    final_linear_layer_weights_.add_grad(grad_final_weights);
}

void VisionTransformer::get_parameters(std::vector<Tensor *> &params)
{
    patch_embedding_.get_parameters(params); // Asume que PatchEmbedding tiene este método
    for (auto &layer : encoder_layers_)
    {
        layer.get_parameters(params); // Asume que EncoderLayer tiene este método
    }
    final_norm_.get_parameters(params); // Asume que LayerNorm tiene este método
    params.push_back(&classification_head_w_);
    params.push_back(&classification_head_b_);
}