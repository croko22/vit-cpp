#include "../../include/model/vit.h"
#include "../../include/core/activation.h"
#include "../../include/core/random.h"
#include <iostream>
#include <algorithm>
#include <fstream>

VisionTransformer::VisionTransformer(int img_size, int patch_sz, int d_mod, int n_layers, int n_classes)
    : image_size(img_size), patch_size(patch_sz), d_model(d_mod),
      num_layers(n_layers), num_classes(n_classes),
      num_patches((img_size / patch_sz) * (img_size / patch_sz)),
      patch_embedding(patch_sz * patch_sz, d_mod),
      class_token(1, d_mod),
      position_embeddings(num_patches + 1, d_mod),
      classification_head(d_mod, n_classes),
      final_ln(d_mod)
{

    for (int i = 0; i < d_model; i++)
    {
        class_token(0, i) = Random::randn(0.0f, 0.01f);
    }
    for (int i = 0; i < position_embeddings.rows; i++)
    {
        for (int j = 0; j < position_embeddings.cols; j++)
        {
            position_embeddings(i, j) = Random::randn(0.0f, 0.01f);
        }
    }

    for (int i = 0; i < num_layers; i++)
    {
        transformer_blocks.push_back(std::make_unique<TransformerBlock>(d_model));
    }
}

Tensor VisionTransformer::image_to_patches(const Tensor &image)
{
    Tensor patches(num_patches, patch_size * patch_size);
    int patches_per_row = image_size / patch_size;
    int patch_idx = 0;
    for (int i = 0; i < patches_per_row; i++)
    {
        for (int j = 0; j < patches_per_row; j++)
        {
            for (int pi = 0; pi < patch_size; pi++)
            {
                for (int pj = 0; pj < patch_size; pj++)
                {
                    int img_row = i * patch_size + pi;
                    int img_col = j * patch_size + pj;
                    patches(patch_idx, pi * patch_size + pj) = image(img_row, img_col);
                }
            }
            patch_idx++;
        }
    }
    return patches;
}

Tensor VisionTransformer::forward(const Tensor &image)
{

    last_patches = image_to_patches(image);

    Tensor patch_emb = patch_embedding.forward(last_patches);

    Tensor sequence = Tensor(num_patches + 1, d_model);
    sequence.set_slice(0, 0, class_token);
    sequence.set_slice(1, 0, patch_emb);

    Tensor current = sequence + position_embeddings;

    for (int i = 0; i < num_layers; i++)
    {
        current = transformer_blocks[i]->forward(current);
    }

    current = final_ln.forward(current);

    Tensor class_token_features = current.slice(0, 1, 0, d_model);

    last_logits = classification_head.forward(class_token_features);

    std::cout << "\nDebug Forward Pass:\n";
    std::cout << "Logits finales: ";
    for (int i = 0; i < num_classes; ++i) 
        std::cout << last_logits(0,i) << " ";
    std::cout << std::endl;
    
    return last_logits;
}

void VisionTransformer::backward(int true_label)
{

    Tensor grad_logits = Activation::softmax(this->last_logits);
    grad_logits(0, true_label) -= 1.0f;

    Tensor grad_class_token_features = classification_head.backward(grad_logits);

    Tensor grad_sequence_after_final_ln(num_patches + 1, d_model);
    grad_sequence_after_final_ln.zero();
    grad_sequence_after_final_ln.set_slice(0, 0, grad_class_token_features);

    Tensor grad_before_final_ln = final_ln.backward(grad_sequence_after_final_ln);

    Tensor grad_current_block_input = grad_before_final_ln;
    for (int i = num_layers - 1; i >= 0; i--)
    {
        grad_current_block_input = transformer_blocks[i]->backward(grad_current_block_input);
    }

    Tensor grad_patch_emb_input = grad_current_block_input.slice(1, num_patches + 1, 0, d_model);

    patch_embedding.backward(grad_patch_emb_input);
}

float VisionTransformer::compute_loss(const Tensor &logits, int true_label)
{
    Tensor probs = Activation::softmax(logits);
    return -log(std::max(probs(0, true_label), 1e-8f));
}

void VisionTransformer::update_weights(float lr)
{
    patch_embedding.update(lr);
    classification_head.update(lr);
    final_ln.update(lr);
    for (auto &block : transformer_blocks)
    {
        block->update(lr);
    }
}

void VisionTransformer::zero_grad()
{
    patch_embedding.zero_grad();
    classification_head.zero_grad();
    final_ln.zero_grad();
    for (auto &block : transformer_blocks)
    {
        block->zero_grad();
    }
}

int VisionTransformer::predict(const Tensor &image)
{
    Tensor logits = forward(image);
    int predicted_class = 0;
    float max_logit = logits(0, 0);
    for (int i = 1; i < num_classes; i++)
    {
        if (logits(0, i) > max_logit)
        {
            max_logit = logits(0, i);
            predicted_class = i;
        }
    }
    return predicted_class;
}

void save_tensor_data(std::ostream &os, const std::string &name, const Tensor &tensor)
{
    os << name << " " << tensor.rows << " " << tensor.cols << std::endl;
    for (int i = 0; i < tensor.rows; ++i)
    {
        for (int j = 0; j < tensor.cols; ++j)
        {
            os << tensor(i, j) << (j == tensor.cols - 1 ? "" : " ");
        }
        os << std::endl;
    }
}

void load_tensor_data(std::istream &is, const std::string &expected_name, Tensor &tensor)
{
    std::string name;
    int rows, cols;
    is >> name >> rows >> cols;
    if (name != expected_name)
    {
        std::cerr << "Error de carga: Nombre de tensor esperado '" << expected_name
                  << "' pero se encontró '" << name << "'" << std::endl;

        return;
    }

    tensor = Tensor(rows, cols);
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            is >> tensor(i, j);
        }
    }
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

        save_tensor_data(ofs, block_prefix + "_attention_proj_weights", transformer_blocks[i]->attention_proj.weight);
        save_tensor_data(ofs, block_prefix + "_attention_proj_biases", transformer_blocks[i]->attention_proj.bias);

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
    class_token = Tensor(1, d_model);
    position_embeddings = Tensor(num_patches + 1, d_model);
    classification_head = Linear(d_model, num_classes);
    final_ln = LayerNorm(d_model);

    transformer_blocks.clear();
    for (int i = 0; i < num_layers; ++i)
    {
        transformer_blocks.push_back(std::make_unique<TransformerBlock>(d_model));
    }

    load_tensor_data(ifs, "class_token", class_token);
    load_tensor_data(ifs, "position_embeddings", position_embeddings);

    load_tensor_data(ifs, "patch_embedding_weights", patch_embedding.weight);
    load_tensor_data(ifs, "patch_embedding_biases", patch_embedding.bias);

    for (int i = 0; i < num_layers; ++i)
    {
        std::string block_prefix = "transformer_block_" + std::to_string(i);
        load_tensor_data(ifs, block_prefix + "_attention_proj_weights", transformer_blocks[i]->attention_proj.weight);
        load_tensor_data(ifs, block_prefix + "_attention_proj_biases", transformer_blocks[i]->attention_proj.bias);

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
