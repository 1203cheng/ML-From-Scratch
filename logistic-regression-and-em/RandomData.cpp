#include <utility>
#include <random>
#include <cmath>
#include "Matrix.h"
using namespace std;

// Uniform distribution generator
random_device rd;
mt19937 generator(rd());
uniform_real_distribution<long double> unif(-1.0, 1.0);
uniform_real_distribution<long double> prob(0.25, 0.75);

long double RandomProbability() {
    return prob(generator);
}

long double GaussianDataGenerator(const long double &mu, const long double &sigma) {
    static bool data = false;
    static long double gaussian_value2;
    
    if (data) {
        data = false;
        return sigma * gaussian_value2 + mu;
    }

    long double gaussian_value1, s, u, v;
    do {
        u = unif(generator);
        v = unif(generator);
        s = u * u + v * v;
    } while (s > 1 || s == 0);
    
    long double mag = sqrt((long double)-2.0 * log(s) / s);
    gaussian_value1 = u * mag;
    gaussian_value2 = v * mag;    
    data = true;

    return sigma * gaussian_value1 + mu;
}

pair<long double, long double> GaussianLinearData(const int &n, long double &sigma,
                                                  const Matrix &W) {
    long double x = unif(generator);
    long double y = 0.0;
    long double base = 1.0;
    for (int i = 0; i != n; ++i) {
        y += (W[i][0] * base);
        base *= x;
    }
    y += GaussianDataGenerator(0, sigma);

    return make_pair(x, y);
}
