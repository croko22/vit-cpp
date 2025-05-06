# Transformer de traducción

Este proyecto implementa un modelo **Transformer** utilizando únicamente **CUDA y C++**, sin bibliotecas de alto nivel como PyTorch o TensorFlow. La implementación es educativa y busca comprender los fundamentos del paper ["Attention is All You Need" (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) a bajo nivel, acelerado en GPU.

## 🧠 Objetivo

Construir un modelo Transformer funcional en CUDA, desarrollando cada componente desde cero:
- Multi-Head Self-Attention
- Positional Encoding
- Feed Forward Networks
- Normalización
- Mecanismos de codificador-decodificador
- Tokenización simple

## ⚙️ Estructura del Proyecto

```

marian-transformer-cuda/
├── attention.cu        # Self-attention con matmul y softmax
├── encoder.cu          # Encoder completo (por implementar)
├── decoder.cu          # Decoder completo (por implementar)
├── embeddings.cu       # Word + positional embeddings
├── utils.cu/.h         # Funciones auxiliares (allocación, IO, etc)
├── data/               # Datos de entrenamiento (WMT14 EN-FR preprocesado)
├── main.cu             # Entry point, ejecución de training/inferencia
└── README.md

````

## 🚀 Cómo compilar y ejecutar

```bash
nvcc -arch=sm_60 -std=c++17 main.cu attention.cu encoder.cu decoder.cu -o transformer
./transformer
````

> Requiere: CUDA Toolkit ≥ 11.0, GPU NVIDIA con Compute Capability ≥ 6.0

## 📚 Dataset

Se usará el dataset **WMT 2014 English-French**.
Por simplicidad, el pipeline usa una versión tokenizada y numerizada (`.ids`). Se recomienda preprocesar con `sentencepiece` o similar en Python y exportar a texto plano.

## 🧩 Estado del Proyecto

* [x] MatMul en CUDA
* [x] Softmax fila a fila
* [x] Cálculo de atención (Q·K^T → softmax → ·V)
* [ ] Multi-head attention
* [ ] Positional encoding
* [ ] Encoder y decoder stacks
* [ ] Entrenamiento simple (cross entropy)
* [ ] Inference con greedy decoding