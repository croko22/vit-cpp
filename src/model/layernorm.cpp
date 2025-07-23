#include "../../include/model/layernorm.h"

LayerNorm::LayerNorm(int d_mod) : d_model(d_mod), eps(1e-5f),
                                  gamma(1, d_mod), beta(1, d_mod),
                                  gamma_grad(1, d_mod), beta_grad(1, d_mod)
{
    for (int i = 0; i < d_model; i++)
    {
        gamma(0, i) = 1.0f;
        beta(0, i) = 0.0f;
    }
}

Tensor LayerNorm::forward(const Tensor &input)
{
    last_input = input;
    last_mean = Tensor(input.rows, 1);
    last_var = Tensor(input.rows, 1);
    Tensor result(input.rows, input.cols);
    for (int i = 0; i < input.rows; i++)
    {
        float mean = 0.0f;
        for (int j = 0; j < input.cols; j++)
        {
            mean += input(i, j);
        }
        mean /= input.cols;
        last_mean(i, 0) = mean;

        float var = 0.0f;
        for (int j = 0; j < input.cols; j++)
        {
            float diff = input(i, j) - mean;
            var += diff * diff;
        }
        var /= input.cols;
        last_var(i, 0) = var;

        for (int j = 0; j < input.cols; j++)
        {
            float normalized = (input(i, j) - mean) / sqrt(var + eps);
            result(i, j) = gamma(0, j) * normalized + beta(0, j);
        }
    }
    return result;
}

Tensor LayerNorm::backward(const Tensor& grad_output)
{
    //TO CHANGE: BORRAR O DEJAR COMO ESTABA
    const Tensor& x = last_input;  // Guardado durante forward()
    const int m = x.cols;          // Dimensión del modelo (d_model)
    
    Tensor grad_input(x.rows, x.cols);
    grad_input.zero();

    for (int i = 0; i < x.rows; ++i) {
        // 1. Recupera mean/var guardados en forward()
        float mean = last_mean(i, 0);
        float var = last_var(i, 0);
        float std_inv = 1.0f / sqrt(var + eps);

        // 2. Gradiente respecto a gamma y beta (acumula sobre todas las filas)
        for (int j = 0; j < m; ++j) {
            float x_hat = (x(i, j) - mean) * std_inv;
            gamma_grad(0, j) += grad_output(i, j) * x_hat;
            beta_grad(0, j) += grad_output(i, j);
        }

        // 3. Gradiente respecto a la entrada (x)
        float sum1 = 0.0f, sum2 = 0.0f;
        for (int j = 0; j < m; ++j) {
            sum1 += grad_output(i, j) * gamma(0, j);
            sum2 += grad_output(i, j) * gamma(0, j) * (x(i, j) - mean);
        }
        
        for (int j = 0; j < m; ++j) {
            float dx_hat = grad_output(i, j) * gamma(0, j);
            float dvar = sum2 * -0.5f * std::pow(var + eps, -1.5f);
            float dmean = sum1 * (-std_inv) + dvar * (-2.0f / m) * (x(i, j) - mean);
            grad_input(i, j) = dx_hat * std_inv + dmean / m + dvar * 2.0f * (x(i, j) - mean) / m;
        }
    }

    return grad_input;  // ¡Este gradiente debe propagarse hacia atrás!
}

void LayerNorm::update(float lr)
{
    // TO CHANGE: Borrar o dejar como estaba
    // zero_grad();
    for (int j = 0; j < d_model; ++j)
    {
        gamma(0, j) -= gamma_grad(0, j) * lr;
        beta(0, j) -= beta_grad(0, j) * lr;
    }
}

void LayerNorm::zero_grad()
{
    gamma_grad.zero();
    beta_grad.zero();
}
