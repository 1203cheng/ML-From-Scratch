#include <stdio.h>
#include <vector>
#include <utility>
#include <random>
#include <cmath>
#include <cassert>
#include <cstring>
#include "header.h"
#include "Matrix.h"
using namespace std;

// Uniform distribution generator
random_device rd;
mt19937 generator(rd());
uniform_real_distribution<long double> unif(-1.0, 1.0);

// 1.a. Univariate gaussian data generator
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

// 1.b. Polynomial basis linear model data generator
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


// 2. Sequential Estimator
Data::Data(int n, long double mean, long double M2) : n(n), mean(mean), M2(M2) {}
void Data::AddData(const long double &v) {
    long double delta = v - mean;
    ++n;
    mean += (delta / n);
    long double delta2 = v - mean;
    M2 += (delta * delta2);

    return;
}

int Data::get_n() {return n;}
long double Data::get_mean() {return mean;}
long double Data::get_variance() {return n > 1 ? M2 / n : 0;}
long double Data::get_sd() {return n > 1 ? sqrt(M2 / n) : 0;}


// Printing result of adding data point
void AddingDataPoint(Data &data, const long double &v) {
    return;
}


// 3. Baysian Linear Regression
Matrix DesignMatrix(const int &n, const long double &x) {
    Matrix ret(1, n);
    long double base = 1;
    for (int j = 0; j != n; ++j) {
        ret[0][j] = base;
        base *= x;
    }

    return ret;
}

void PrintPosterior(const Matrix &Mu, const Matrix &Lambda) {
    int n = Mu.size();
    printf("Posterior mean:\n");
    for (int i = 0; i != n; ++i) {
        printf("\t%.10Lf\n", Mu[i][0]);
    }

    printf("\nPosterior variance:\n");
    for (int i = 0; i != n; ++i) {
        for (int j = 0; j != n; ++j) {
            printf("%15.10Lf,", Lambda[i][j]);
        }
        printf("\n");
    }
}

int SaveMatrix(FILE *fptr, const string &title, const Matrix &mat) {
    int n = mat.size();
    int m = mat[0].size();
    if (fprintf(fptr, "%s", title.c_str()) < 0) {return -1;}
    for (int i = 0; i != n; ++i) {
        for (int j = 0; j != m; ++j) {
            fprintf(fptr, "%.16Lf,", mat[i][j]);
        }
        fprintf(fptr, "\n");
    }
    fprintf(fptr, "\n");

    return 0;
}