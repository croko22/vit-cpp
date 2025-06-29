#ifndef TRANSFORMER_HPP
#define TRANSFORMER_HPP

#include "encoder.hpp"
#include "decoder.hpp"
#include "../core/tensor.hpp"
#include <vector>

class Transformer
{
private:
    int num_layers_;
    int d_model_;

    std::vector<EncoderLayer> encoder_layers_;
    std::vector<DecoderLayer> decoder_layers_;
    Tensor final_linear_layer_weights_;

public:
    // El constructor ya no necesita vocab_size ni max_seq_len
    Transformer(int num_layers, int d_model, int num_heads, int d_ff, int target_vocab_size);

    /**
     * @brief El paso forward completo del modelo Transformer.
     * @param src_tensor Tensor pre-procesado de la secuencia de entrada [src_len, d_model].
     * @param tgt_tensor Tensor pre-procesado de la secuencia de salida [tgt_len, d_model].
     * @return Tensor de logits (sin softmax) con forma [tgt_len, target_vocab_size].
     */
    Tensor forward(const Tensor &src_tensor, const Tensor &tgt_tensor);

private:
    Tensor encode(const Tensor &src_tensor);
    Tensor decode(const Tensor &tgt_tensor, const Tensor &encoder_output);
    static Tensor create_look_ahead_mask(int size);
};

#endif // TRANSFORMER_HPP