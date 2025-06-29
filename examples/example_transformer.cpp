#include "../include/model/transformer.hpp"
#include <iostream>
#include <vector>

Tensor create_dummy_tensor(const std::vector<int> &shape, float value)
{
    Tensor t(shape);
    std::vector<float> data(t.get_size(), value);
    t.from_vector(data);
    return t;
}

int main()
{
    std::cout << "----------------------------------------------------" << std::endl;
    std::cout << "DEMOSTRACION DEL FORWARD PASS DE UN TRANSFORMER EN C++ PURO" << std::endl;
    std::cout << "----------------------------------------------------\n"
              << std::endl;

    int num_layers = 6;
    int d_model = 512;
    int num_heads = 8;
    int d_ff = 2048;
    int target_vocab_size = 1000;

    std::cout << "[INFO] Creando instancia del modelo Transformer..." << std::endl;
    std::cout << "       - Capas: " << num_layers << std::endl;
    std::cout << "       - Dimension del modelo (d_model): " << d_model << std::endl;
    std::cout << "       - Cabezales de atencion (num_heads): " << num_heads << std::endl;

    Transformer model(num_layers, d_model, num_heads, d_ff, target_vocab_size);
    std::cout << "[SUCCESS] Modelo creado.\n"
              << std::endl;

    int src_seq_len = 10;
    int tgt_seq_len = 15;
    // int batch_size = 1;

    std::cout << "[INFO] Creando tensores de entrada falsos (dummy data)..." << std::endl;

    Tensor src_tensor = create_dummy_tensor({src_seq_len, d_model}, 1.0f);

    Tensor tgt_tensor = create_dummy_tensor({tgt_seq_len, d_model}, 2.0f);
    std::cout << "       - Tensor de Encoder (fuente): [" << src_tensor.get_shape()[0] << ", " << src_tensor.get_shape()[1] << "]" << std::endl;
    std::cout << "       - Tensor de Decoder (destino): [" << tgt_tensor.get_shape()[0] << ", " << tgt_tensor.get_shape()[1] << "]\n"
              << std::endl;

    std::cout << "[INFO] Ejecutando model.forward()..." << std::endl;
    std::cout << "       Esto pasara los datos a traves de todas las capas del Encoder y Decoder." << std::endl;

    Tensor output_logits = model.forward(src_tensor, tgt_tensor);

    std::cout << "[SUCCESS] Forward Pass completado!\n"
              << std::endl;

    std::cout << "[INFO] Verificando la forma del tensor de salida..." << std::endl;
    const auto &output_shape = output_logits.get_shape();

    std::cout << "       - Forma Esperada: [" << tgt_seq_len << ", " << target_vocab_size << "]" << std::endl;
    std::cout << "       - Forma Obtenida: [" << output_shape[0] << ", " << output_shape[1] << "]" << std::endl;

    if (output_shape[0] == tgt_seq_len && output_shape[1] == target_vocab_size)
    {
        std::cout << "[SUCCESS] ¡La forma de la salida es correcta!\n"
                  << std::endl;
    }
    else
    {
        std::cout << "[ERROR] ¡La forma de la salida es INCORRECTA!\n"
                  << std::endl;
        return 1;
    }

    std::cout << "[INFO] Mostrando los primeros 5 valores de los logits de salida:" << std::endl;
    output_logits.print();

    std::cout << "\n----------------------------------------------------" << std::endl;
    std::cout << "DEMOSTRACION FINALIZADA CON EXITO." << std::endl;
    std::cout << "----------------------------------------------------" << std::endl;

    return 0;
}