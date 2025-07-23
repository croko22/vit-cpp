#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "tensor.h"
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class Activation
{
public:
    static float relu(float x);
    static float relu_derivative(float x);
    static float gelu(float x);
    static float gelu_derivative(float x);
    static Tensor apply(const Tensor &input, float (*func)(float));
    static Tensor softmax(const Tensor &input);
    static Tensor softmax_grad(const Tensor& logits, int true_class);
};

#endif // ACTIVATION_H