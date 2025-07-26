#include "../../include/model/vit.h"
#include "../../include/model/encoder.h"
#include "../../include/core/activation.h"
#include "../../include/core/random.h"
#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

void save_tensor_data(std::ostream &os, const std::string &name, const Tensor &tensor)
{
    const auto &shape = tensor.get_shape();
    os << name << " " << shape.size();
    for (int dim : shape)
    {
        os << " " << dim;
    }
    os << std::endl;
    const auto &data = tensor.get_data();
    for (size_t i = 0; i < data.size(); ++i)
    {
        os << data[i] << (i == data.size() - 1 ? "" : " ");
    }
    os << std::endl;
}

void load_tensor_data(std::istream &is, const std::string &expected_name, Tensor &tensor)
{
    std::string name;
    int num_dims;
    is >> name >> num_dims;
    if (name != expected_name)
    {
        throw std::runtime_error("Error de carga: Se esperaba '" + expected_name + "' pero se encontró '" + name + "'");
    }
    std::vector<int> shape(num_dims);
    for (int i = 0; i < num_dims; ++i)
    {
        is >> shape[i];
    }
    tensor = Tensor(shape);
    auto &data = tensor.get_data();
    for (size_t i = 0; i < data.size(); ++i)
    {
        is >> data[i];
    }
}

VisionTransformer::VisionTransformer(int img_size, int patch_sz, int d_mod, int n_layers, int n_classes)
    : image_size(img_size),
      patch_size(patch_sz),
      d_model(d_mod),
      num_layers(n_layers),
      num_classes(n_classes),
      num_patches((img_size / patch_sz) * (img_size / patch_sz)),
      patch_embedding(patch_sz * patch_sz * 1, d_mod),

      class_token({1, d_mod}),
      position_embeddings({(num_patches + 1), d_mod}),
      class_token_grad({1, d_mod}),
      position_embeddings_grad({(num_patches + 1), d_mod}),

      classification_head(d_mod, n_classes),
      final_ln(d_mod)
{

    Tensor::xavier_init(this->class_token);
    Tensor::xavier_init(this->position_embeddings);

    int num_heads = 8;
    int d_ff = d_model * 4;
    for (int i = 0; i < num_layers; i++)
    {
        transformer_blocks.push_back(std::make_unique<TransformerBlock>(d_model, num_heads, d_ff));
    }

    class_token_grad.zero();
    position_embeddings_grad.zero();
}

Tensor VisionTransformer::image_to_patches(const Tensor &image)
{
    Tensor patches({num_patches, patch_size * patch_size * 1});
    int patches_per_row = image_size / patch_size;
    int patch_idx = 0;

    for (int i = 0; i < patches_per_row; i++)
    {
        for (int j = 0; j < patches_per_row; j++)
        {
            Tensor patch_2d = image.slice(i * patch_size, (i + 1) * patch_size, j * patch_size, (j + 1) * patch_size);
            patches.set_slice(patch_idx, 0, patch_2d.flatten());
            patch_idx++;
        }
    }
    return patches;
}

Tensor VisionTransformer::forward(const Tensor &image)
{
    last_patches = image_to_patches(image);
    Tensor patch_emb = patch_embedding.forward(last_patches);

    Tensor sequence({num_patches + 1, d_model});
    sequence.set_slice(0, 0, class_token);
    sequence.set_slice(1, 0, patch_emb);

    Tensor current = sequence + position_embeddings;

    for (const auto &block : transformer_blocks)
    {
        current = block->forward(current);
    }

    current = final_ln.forward(current);

    Tensor class_token_features = current.slice(0, 1, 0, d_model);
    last_logits = classification_head.forward(class_token_features);

    return last_logits;
}

// En src/model/vit.cpp

void VisionTransformer::backward(int true_label)
{
    // 1. Calcular el gradiente inicial de la pérdida (Cross-Entropy con Softmax)
    Tensor grad_logits = Activation::softmax(last_logits);
    grad_logits.get_data()[true_label] -= 1.0f; // grad = probabilities - one_hot_label

    // 2. Propagar hacia atrás a través de la capa de clasificación
    Tensor grad_class_token_features = classification_head.backward(grad_logits);

    // 3. Crear el gradiente para la secuencia completa. Solo el token [CLS] recibe gradiente.
    Tensor grad_full_sequence({num_patches + 1, d_model}); // Inicializa a ceros
    grad_full_sequence.set_slice(0, 0, grad_class_token_features);

    // 4. Propagar a través de la capa LayerNorm final
    Tensor grad_current = final_ln.backward(grad_full_sequence);

    // 5. Propagar a través de los bloques Transformer
    for (int i = num_layers - 1; i >= 0; i--)
    {
        grad_current = transformer_blocks[i]->backward(grad_current);
    }
    // 6. AGREGAR: Acumular gradientes para position_embeddings y class_token
    // Los gradientes se acumulan durante el batch
    for (size_t i = 0; i < position_embeddings_grad.get_data().size(); ++i)
    {
        position_embeddings_grad.get_data()[i] += grad_current.get_data()[i];
    }

    // Para class_token (solo la primera fila de grad_current)
    for (size_t i = 0; i < static_cast<size_t>(d_model); ++i)
    {
        class_token_grad.get_data()[i] += grad_current.get_data()[i]; // Primera fila
    }

    // --- CORRECCIÓN ADICIONAL (ver punto 3) ---
    // El gradiente también debe aplicarse a los embeddings de posición
    // position_embeddings.grad += grad_current; // (Necesitarías añadir un tensor .grad a tu clase Tensor)
    // class_token.grad += grad_current.slice(0,1,...);
    // Por ahora, nos centraremos en los errores principales.

    // 6. Extraer el gradiente para los patch embeddings
    Tensor grad_patch_emb = grad_current.slice(1, num_patches + 1, 0, d_model);

    // 7. Propagar a través del embedding de patches
    patch_embedding.backward(grad_patch_emb);
}

float VisionTransformer::compute_loss(const Tensor &logits, int true_label)
{
    Tensor probs = Activation::softmax(logits);
    return -std::log(probs(0, true_label) + 1e-8f);
}

// Reemplazar la función update_weights existente:

void VisionTransformer::update_weights(float lr, int batch_size)
{
    // 1. Actualizar capas con parámetros entrenables
    patch_embedding.update(lr, batch_size);
    classification_head.update(lr, batch_size);
    final_ln.update(lr, batch_size);

    // 2. Actualizar bloques transformer
    for (auto &block : transformer_blocks)
    {
        block->update(lr, batch_size);
    }

    // 3. AGREGAR: Actualizar class_token y position_embeddings
    // Usar los gradientes acumulados para actualizar estos parámetros

    // Para class_token: aplicar gradiente acumulado
    for (size_t i = 0; i < class_token.get_data().size(); ++i)
    {
        class_token.get_data()[i] -= lr * class_token_grad.get_data()[i] / batch_size;
    }

    // Línea 205 - Para position_embeddings: aplicar gradiente acumulado
    for (size_t i = 0; i < position_embeddings.get_data().size(); ++i)
    {
        position_embeddings.get_data()[i] -= lr * position_embeddings_grad.get_data()[i] / batch_size;
    }
}

void VisionTransformer::zero_grad()
{
    // 1. Limpiar gradientes de capas
    patch_embedding.zero_grad();
    classification_head.zero_grad();
    final_ln.zero_grad();

    // 2. Limpiar gradientes de bloques transformer
    for (auto &block : transformer_blocks)
    {
        block->zero_grad();
    }

    // 3. AGREGAR: Limpiar gradientes de embeddings
    class_token_grad.zero();
    position_embeddings_grad.zero();
}

int VisionTransformer::predict(const Tensor &image)
{
    Tensor logits = forward(image);
    return logits.argmax();
}

int VisionTransformer::predictWithLogits(const Tensor &logits)
{
    return logits.argmax();
}

void VisionTransformer::save_model(const std::string &filename) const
{
    std::ofstream ofs(filename);
    if (!ofs.is_open())
    {
        std::cerr << "Error: No se pudo abrir el archivo para guardar el modelo: " << filename << std::endl;
        return;
    }

    ofs << "MODEL_CONFIG" << std::endl;
    ofs << "image_size " << image_size << std::endl;
    ofs << "patch_size " << patch_size << std::endl;
    ofs << "d_model " << d_model << std::endl;
    ofs << "num_layers " << num_layers << std::endl;
    ofs << "num_classes " << num_classes << std::endl;
    ofs << "num_patches " << num_patches << std::endl;

    save_tensor_data(ofs, "class_token", class_token);
    save_tensor_data(ofs, "position_embeddings", position_embeddings);

    save_tensor_data(ofs, "patch_embedding_weights", patch_embedding.weight);
    save_tensor_data(ofs, "patch_embedding_biases", patch_embedding.bias);

    for (int i = 0; i < num_layers; ++i)
    {
        std::string block_prefix = "transformer_block_" + std::to_string(i);

        save_tensor_data(ofs, block_prefix + ".mha.q_proj.weight", transformer_blocks[i]->mha.q_proj.weight);
        save_tensor_data(ofs, block_prefix + ".mha.q_proj.bias", transformer_blocks[i]->mha.q_proj.bias);
        save_tensor_data(ofs, block_prefix + ".mha.k_proj.weight", transformer_blocks[i]->mha.k_proj.weight);
        save_tensor_data(ofs, block_prefix + ".mha.k_proj.bias", transformer_blocks[i]->mha.k_proj.bias);
        save_tensor_data(ofs, block_prefix + ".mha.v_proj.weight", transformer_blocks[i]->mha.v_proj.weight);
        save_tensor_data(ofs, block_prefix + ".mha.v_proj.bias", transformer_blocks[i]->mha.v_proj.bias);
        save_tensor_data(ofs, block_prefix + ".mha.out_proj.weight", transformer_blocks[i]->mha.out_proj.weight);
        save_tensor_data(ofs, block_prefix + ".mha.out_proj.bias", transformer_blocks[i]->mha.out_proj.bias);

        save_tensor_data(ofs, block_prefix + "_mlp_fc1_weights", transformer_blocks[i]->mlp.fc1.weight);
        save_tensor_data(ofs, block_prefix + "_mlp_fc1_biases", transformer_blocks[i]->mlp.fc1.bias);
        save_tensor_data(ofs, block_prefix + "_mlp_fc2_weights", transformer_blocks[i]->mlp.fc2.weight);
        save_tensor_data(ofs, block_prefix + "_mlp_fc2_biases", transformer_blocks[i]->mlp.fc2.bias);

        save_tensor_data(ofs, block_prefix + "_mlp_ln_gamma", transformer_blocks[i]->mlp.ln.gamma);
        save_tensor_data(ofs, block_prefix + "_mlp_ln_beta", transformer_blocks[i]->mlp.ln.beta);
        save_tensor_data(ofs, block_prefix + "_ln1_gamma", transformer_blocks[i]->ln1.gamma);
        save_tensor_data(ofs, block_prefix + "_ln1_beta", transformer_blocks[i]->ln1.beta);
        save_tensor_data(ofs, block_prefix + "_ln2_gamma", transformer_blocks[i]->ln2.gamma);
        save_tensor_data(ofs, block_prefix + "_ln2_beta", transformer_blocks[i]->ln2.beta);
    }

    save_tensor_data(ofs, "classification_head_weights", classification_head.weight);
    save_tensor_data(ofs, "classification_head_biases", classification_head.bias);

    save_tensor_data(ofs, "final_ln_gamma", final_ln.gamma);
    save_tensor_data(ofs, "final_ln_beta", final_ln.beta);

    ofs.close();
    std::cout << "Modelo guardado exitosamente en: " << filename << std::endl;
}

void VisionTransformer::load_model(const std::string &filename)
{
    std::ifstream ifs(filename);
    if (!ifs.is_open())
    {
        std::cerr << "Error: No se pudo abrir el archivo para cargar el modelo: " << filename << std::endl;
        return;
    }

    std::string tag;
    ifs >> tag;
    if (tag != "MODEL_CONFIG")
    {
        std::cerr << "Error de carga: Formato de archivo inesperado. Se esperaba 'MODEL_CONFIG'." << std::endl;
        ifs.close();
        return;
    }

    std::string param_name;
    ifs >> param_name >> image_size;
    ifs >> param_name >> patch_size;
    ifs >> param_name >> d_model;
    ifs >> param_name >> num_layers;
    ifs >> param_name >> num_classes;
    ifs >> param_name >> num_patches;

    patch_embedding = Linear(patch_size * patch_size, d_model);
    class_token = Tensor({1, d_model});
    position_embeddings = Tensor({num_patches + 1, d_model});
    classification_head = Linear(d_model, num_classes);
    final_ln = LayerNorm(d_model);

    transformer_blocks.clear();
    for (int i = 0; i < num_layers; ++i)
    {
        transformer_blocks.push_back(std::make_unique<TransformerBlock>(d_model, 8, d_model * 4));
    }

    load_tensor_data(ifs, "class_token", class_token);
    load_tensor_data(ifs, "position_embeddings", position_embeddings);

    load_tensor_data(ifs, "patch_embedding_weights", patch_embedding.weight);
    load_tensor_data(ifs, "patch_embedding_biases", patch_embedding.bias);

    for (int i = 0; i < num_layers; ++i)
    {
        std::string block_prefix = "transformer_block_" + std::to_string(i);
        load_tensor_data(ifs, block_prefix + ".mha.q_proj.weight", transformer_blocks[i]->mha.q_proj.weight);
        load_tensor_data(ifs, block_prefix + ".mha.q_proj.bias", transformer_blocks[i]->mha.q_proj.bias);
        load_tensor_data(ifs, block_prefix + ".mha.k_proj.weight", transformer_blocks[i]->mha.k_proj.weight);
        load_tensor_data(ifs, block_prefix + ".mha.k_proj.bias", transformer_blocks[i]->mha.k_proj.bias);
        load_tensor_data(ifs, block_prefix + ".mha.v_proj.weight", transformer_blocks[i]->mha.v_proj.weight);
        load_tensor_data(ifs, block_prefix + ".mha.v_proj.bias", transformer_blocks[i]->mha.v_proj.bias);
        load_tensor_data(ifs, block_prefix + ".mha.out_proj.weight", transformer_blocks[i]->mha.out_proj.weight);
        load_tensor_data(ifs, block_prefix + ".mha.out_proj.bias", transformer_blocks[i]->mha.out_proj.bias);

        load_tensor_data(ifs, block_prefix + "_mlp_fc1_weights", transformer_blocks[i]->mlp.fc1.weight);
        load_tensor_data(ifs, block_prefix + "_mlp_fc1_biases", transformer_blocks[i]->mlp.fc1.bias);
        load_tensor_data(ifs, block_prefix + "_mlp_fc2_weights", transformer_blocks[i]->mlp.fc2.weight);
        load_tensor_data(ifs, block_prefix + "_mlp_fc2_biases", transformer_blocks[i]->mlp.fc2.bias);

        load_tensor_data(ifs, block_prefix + "_mlp_ln_gamma", transformer_blocks[i]->mlp.ln.gamma);
        load_tensor_data(ifs, block_prefix + "_mlp_ln_beta", transformer_blocks[i]->mlp.ln.beta);
        load_tensor_data(ifs, block_prefix + "_ln1_gamma", transformer_blocks[i]->ln1.gamma);
        load_tensor_data(ifs, block_prefix + "_ln1_beta", transformer_blocks[i]->ln1.beta);
        load_tensor_data(ifs, block_prefix + "_ln2_gamma", transformer_blocks[i]->ln2.gamma);
        load_tensor_data(ifs, block_prefix + "_ln2_beta", transformer_blocks[i]->ln2.beta);
    }

    load_tensor_data(ifs, "classification_head_weights", classification_head.weight);
    load_tensor_data(ifs, "classification_head_biases", classification_head.bias);

    load_tensor_data(ifs, "final_ln_gamma", final_ln.gamma);
    load_tensor_data(ifs, "final_ln_beta", final_ln.beta);

    ifs.close();
    std::cout << "Modelo cargado exitosamente desde: " << filename << std::endl;
}
