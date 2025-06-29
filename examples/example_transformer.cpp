#include "encoder.hpp"
#include <iostream>

int main()
{
    const int d_model = 512;
    const int num_heads = 8;
    const int dff = 2048;

    EncoderBlock enc_block(d_model, num_heads, dff);

    float input[d_model] = {0};
    float output[d_model];

    enc_block.forward(input, output);

    std::cout << "Salida del Encoder Block:\n";
    for (int i = 0; i < 10; ++i)
        std::cout << output[i] << " ";
    std::cout << "\n";

    return 0;
}