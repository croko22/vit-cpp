#include "../../include/optimizer/adamw.h"
#include <cmath>

AdamW::AdamW(std::vector<Parameter> params, float lr, float b1, float b2, float eps, float wd)
    : Optimizer(params, lr), beta1(b1), beta2(b2), epsilon(eps), weight_decay(wd), t(0)
{

    for (const auto &p : this->params)
    {
        m.emplace_back(p.weight->get_shape());
        v.emplace_back(p.weight->get_shape());
        m.back().zero();
        v.back().zero();
    }
}

void AdamW::step()
{
    t++;

    for (size_t i = 0; i < params.size(); ++i)
    {
        Tensor *weight = params[i].weight;
        Tensor *grad = params[i].grad;

        for (int j = 0; j < weight->get_size(); ++j)
        {

            m[i].data[j] = beta1 * m[i].data[j] + (1 - beta1) * grad->data[j];

            v[i].data[j] = beta2 * v[i].data[j] + (1 - beta2) * grad->data[j] * grad->data[j];

            float m_hat = m[i].data[j] / (1 - std::pow(beta1, t));
            float v_hat = v[i].data[j] / (1 - std::pow(beta2, t));

            float adam_update = lr * m_hat / (std::sqrt(v_hat) + epsilon);

            float decay = lr * weight_decay * weight->data[j];

            weight->data[j] -= (adam_update + decay);
        }
    }
}