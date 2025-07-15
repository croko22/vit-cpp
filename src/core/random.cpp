#include "../../include/core/random.h"
#include <random>

std::mt19937 Random::gen;
std::random_device Random::rd;

void Random::seed(unsigned int s)
{
    if (s == 0)
        gen.seed(rd());
    else
        gen.seed(s);
}

float Random::randn(float mean, float stddev)
{
    std::normal_distribution<float> d(mean, stddev);
    return d(gen);
}

float Random::uniform(float min, float max)
{
    std::uniform_real_distribution<float> d(min, max);
    return d(gen);
}

int Random::randint(int min, int max)
{
    std::uniform_int_distribution<int> d(min, max);
    return d(gen);
}
