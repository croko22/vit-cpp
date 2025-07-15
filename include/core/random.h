#ifndef RANDOM_H
#define RANDOM_H

#include <random>

class Random
{
public:
    static std::mt19937 gen;
    static std::random_device rd;
    static void seed(unsigned int s = 0);
    static float randn(float mean = 0.0f, float stddev = 1.0f);
    static float uniform(float min = 0.0f, float max = 1.0f);
    static int randint(int min, int max);
};

#endif // RANDOM_H