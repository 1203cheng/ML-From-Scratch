#include <stdio.h>
#include <cctype>
#include <cmath>
#include <vector>
#include <utility>
#include <string>
#include "header.h"
#include "Matrix.h"
using namespace std;

void printUsage() {
    printf("Usage: ./program <part>\n");
    printf("<part> = \n1: run part 2. and 3.\n2: run part 2.\n3: run part 3.\n");
    return;
}

int part2() {
    printf("================2. Sequential Estimator================\n");
    const long double ERROR = 1e-4;
    long double mu = 3.0;
    long double variance = 5.0;
    printf("m = ");
    scanf("%Lf", &mu);
    printf("s = ");
    scanf("%Lf", &variance);
    long double sigma = sqrt(variance);
    printf("Data point source function: N(%.2Lf, %.2Lf)\n\n", mu, variance);

    int n_data = 0;
    long double prev_mu = 0;
    long double prev_var = 0;
    Data data;
    while (true) {
        long double new_data = GaussianDataGenerator(mu, sigma);
        printf("Add data point %.15Lf\n", new_data);

        data.AddData(new_data);
        ++n_data;
        long double cur_mu = data.get_mean();
        long double cur_var = data.get_variance();
        printf("Mean = %-20.15LfVariance = %.15Lf\n", cur_mu, cur_var);

        if (n_data > 30 && (abs(cur_mu - prev_mu) < ERROR) && (abs(cur_var - prev_var)) < ERROR) {
            printf("\nTotal number of data generated: %d\n", n_data);
            break;
        }

        prev_mu = cur_mu;
        prev_var = cur_var;
    }

    return 0;
}

int part3() {
    printf("================3. Baysian Linear Regression================\n");
    const long double ERROR = 1e-6;
    long double a, b;
    int n;
    printf("a = ");
    scanf("%Lf", &a);
    printf("b = ");
    scanf("%Lf", &b);
    printf("n = ");
    scanf("%d", &n);
    long double sigma = sqrt(a);
    long double a_i = 1.0 / ((a != 0) ? a : 1e-8);
    
    Matrix w(n, 1);
    printf("w(format: <w1 w2 ... wn>) = ");
    for (int i = 0; i != n; ++i) {
        scanf("%Lf", &w[i][0]);
    }

    Matrix B(n, n, b);
    long double pred_mean[2], pred_var[2];
    vector<pair<Matrix, Matrix>> Lambda(2, {B, B});
    vector<Matrix> Mu(2, Matrix(n, 1, 0)); 
    vector<long double> x(0);
    vector<long double> y(0);
    pair<Matrix, Matrix> L_store;
    pair<Matrix, Matrix> M_store;
    Matrix X;
    int prior, post;
    const int CAP = 1e5;
    int i = 0;
    for (; i != CAP; ++i) {
        if (i == 10) {
            L_store.first = Lambda[post].second;
            M_store.first = Mu[post];
        }
        else if (i == 50) {
            L_store.second = Lambda[post].second;
            M_store.second = Mu[post];
        }

        prior = i & 1;
        post = (i + 1) & 1;

        auto [pos_x, pos_y] = GaussianLinearData(n, sigma, w);
        printf("Add data point (%LF, %LF)\n\n", pos_x, pos_y);
        X = DesignMatrix(n, pos_x);
        x.emplace_back(pos_x);
        y.emplace_back(pos_y);
        
        Lambda[post].first = a_i * (Transpose(X) * X) + Lambda[prior].first;
        Lambda[post].second = Inverse(Lambda[post].first);
        Mu[post] = Lambda[post].second *
                   (a_i * Transpose(X) * y[i] + Lambda[prior].first * Mu[prior]);
        PrintPosterior(Mu[post], Lambda[post].second);

        pred_mean[post] = (X * Mu[post])[0].front();
        pred_var[post] = (X * Lambda[post].second * Transpose(X))[0].front() + a;
        printf("Predictive distribution ~ N(%Lf, %Lf)\n\n", pred_mean[post], pred_var[post]);

        if (i > 50 && abs(pred_var[0] - pred_var[1]) < ERROR) {
            ++i;
            break;
        }
    }
    printf("Total number of data generated: %d\n", i + 1);

    // Saving results to "result.csv"
    FILE *fptr = fopen("result.csv", "w");
    fprintf(fptr, "n,%d\na,%.16Lf\nb,%.16Lf\n", n, a, b);
    fprintf(fptr, "w:\n");
    for (i = 0; i != n; ++i) {
        fprintf(fptr, "%.16Lf,", w[i][0]);
    }
    fprintf(fptr, "\n\n");

    if (SaveMatrix(fptr, "Mu 10:\n", M_store.first) < 0) {return -1;}
    if (SaveMatrix(fptr, "Mu 50:\n", M_store.second) < 0) {return -1;}
    if (SaveMatrix(fptr, "Mu:\n", Mu[post]) < 0) {return -1;}
    if (SaveMatrix(fptr, "Lambda_i 10:\n", L_store.first) < 0) {return -1;}
    if (SaveMatrix(fptr, "Lambda_i 50:\n", L_store.second) < 0) {return -1;}
    if (SaveMatrix(fptr, "Lambda_i:\n", Lambda[post].second) < 0) {return -1;}
    
    // Generate the curve
    long double pos_x;
    long double step = (2.0 - (-2.0)) / (29); 
    fprintf(fptr, "Design matrix X:\n");
    for (int i = 0; i < 30; ++i) {
        pos_x = -2.0 + step * i; 
        Matrix X(DesignMatrix(n, pos_x));
        for (long double e : X[0]) {
            fprintf(fptr, "%.16Lf,", e);
        }
        fprintf(fptr, "\n");
    }
    fprintf(fptr, "\n");

    fprintf(fptr, "Data:\n");
    int data_n = x.size();
    for (i = 0; i != data_n; ++i) {
        fprintf(fptr, "%.16Lf,%.16Lf\n", x[i], y[i]);
    }

    fclose(fptr);
    return 0;
}

int main(int argc, char *argv[]) {
    // Deciding running part 2. or 3.
    if (argc == 1 || !isdigit(*argv[1])) {
        printUsage();        
        return 0;
    }

    switch (stoi(argv[1])) {
        case 2: part2(); break;
        case 1: part2();
        case 3: part3(); break;
        default: printUsage(); break;
    }

    return 0;
}