# Transformer de traducción

Este proyecto implementa un modelo **Transformer** utilizando únicamente **CUDA y C++**, sin bibliotecas de alto nivel como PyTorch o TensorFlow. La implementación es educativa y busca comprender los fundamentos del paper ["Attention is All You Need" (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) a bajo nivel, acelerado en GPU.

## Notebooks
- [TIA - MARIAN | Transformer_Trainer_Notebook](https://colab.research.google.com/drive/134n_xEv7VfA2_5VniJEgzhNcSh8etRPz#scrollTo=VmYGVziz50tu)
- [TIA - MARIAN | input_embeddings](https://colab.research.google.com/drive/12Tq-RRQ8HntnFcKtEui3OznjHztWN92q?usp=sharing)

## 🧠 Objetivo

Construir un modelo Transformer funcional en CUDA, desarrollando cada componente desde cero:
- Multi-Head Self-Attention
- Positional Encoding
- Feed Forward Networks
- Normalización
- Mecanismos de codificador-decodificador
- Tokenización simple

<!-- ## ⚙️ Estructura del Proyecto

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
``` -->

## 🚀 Cómo compilar y ejecutar

GPU
```bash
nvcc -arch=sm_60 -std=c++17 main.cu attention.cu encoder.cu decoder.cu -o transformer
./transformer
```

CPU
´´´
make run
´´´