#define NDEBUG
#include <cstdio>
#include <cctype>
#include <string>
#include <limits>
#include <cassert>
#include <utility>
#include <vector>
#include <cmath>
#include <bitset>
#include "Matrix.h"
#include "RandomData.h"
using namespace std;

const long double ERROR = 1e-4;
const long double MOMENTUM = 0.9;
const long double INF = numeric_limits<long double>::max();
const long double INF_INT = numeric_limits<int>::max();

enum DescentMethod {GradientDescent, NewtonsMethod};

Matrix Sigmoid(const Matrix &X) {
    Matrix ret = -X;
    for (auto &row : ret) {
        for (auto &e : row) {
            e = 1 / (1 + exp(e));
        }
    }

    return ret;
}

Matrix DSigmoid(const Matrix &X) {
    Matrix ret = -X;
    for (auto &row : ret) {
        for (auto &e : row) {
            e = exp(e);
            e = e / pow(1 + e, 2);
        }
    }

    return ret;
}

bool isInvertible(const Matrix &m) {
    if (m.size() != 3 || m[0].size() != 3) {return false;}
    
    long double tmp = m[0][0] * m[1][1] * m[2][2] + m[0][1] * m[1][2] * m[2][0];
    tmp += (m[0][2] * m[1][0] * m[2][1] - m[0][2] * m[1][1] * m[2][0]);
    tmp -= (m[0][1] * m[1][0] * m[2][2] + m[0][0] * m[1][2] * m[2][1]);
    if (tmp == 0.0) {return false;}

    return true;
}

Matrix gradient(const Matrix &X, const Matrix &XT, const Matrix &t, const Matrix &w) {
    assert(X.size() == XT[0].size() && X[0].size() == XT.size() && X.size() == t.size()
           && XT.size() == w.size() && w[0].size() == 1 && t[0].size() == 1);
    Matrix ret = Sigmoid(X * w);
    ret = XT * (t - ret);
    
    return ret;
}

Matrix Hessian(const Matrix &X, const Matrix &XT ,const Matrix &w) {
    assert(X.size() == XT[0].size() && X[0].size() == XT.size() && XT.size() == w.size()
           && w[0].size() == 1);  

    int n = X.size(); 
    Matrix tmp = DSigmoid(X * w);
    Matrix ret(n, n);
    for (int i = 0; i != n; ++i) {
        ret[i][i] = tmp[i][0];
    }
    ret = XT * ret * X;

    return ret;
}

int EvaluateAccuracy(const Matrix &t, const Matrix &y) {
    int n = t.size();
    assert(n == y.size() && t[0].size() == 1 && y[0].size() == 1);
    int TP = 0, FP = 0, TN = 0, FN = 0;

    for (int i = 0; i != n; ++i) {
        if (y[i][0] == 1) {
            (y[i][0] == t[i][0]) ? TP += 1 : FP += 1;
        }
        else {
            (y[i][0] == t[i][0]) ? TN += 1 : FN += 1;
        }
    }

    // Write to result.csv
    FILE *fptr = fopen("result.csv", "a");
    if (fptr == nullptr) {
        perror("Failed to open result.csv");
        return -1;
    }

    fprintf(fptr, "\nStatistics:\nTP:\n%d\nFP:\n%d\nTN:\n%d\nFN:\n%d\n", TP, FP, TN, FN);

    fclose(fptr);
    return 0;
}

int Descent(const Matrix &X, const Matrix &t, int method, long double eta = 0.01) {
    assert(X.size() == t.size() && t[0].size() == 1);
    int n = X.size();

    // Find w_MLE
    Matrix XT = Transpose(X);
    int w_n = X[0].size();
    Matrix w(w_n, 1);
    vector<Matrix> v(2, Matrix(w_n, 1));

    int cur = 0, pre = (~cur) & 1;
    v[cur] = Matrix(w_n, 1, INF);
    auto isConverged = [&]() {
        for (int i = 0; i != w_n; ++i) {
            if (abs(w[i][0] * ERROR) < abs(v[cur][i][0])) {return false;}
        }
        return true;
    };

    for (int i = 0; !isConverged(); ++i) {
        cur = i & 1;
        pre = (~i) & 1;
        switch (method) {
            case GradientDescent: {
                v[cur] = gradient(X, XT, t, w);
                break;
            }
            case NewtonsMethod: {
                v[cur] = n * Hessian(X, XT, w);
                if (!isInvertible(v[cur])) {
                    v[cur] = n * gradient(X, XT, t, w);
                }
                else {
                    v[cur] = Inverse(v[cur]) * (n * gradient(X, XT, t, w));
                }
                break;
            }
        }
        v[cur] += MOMENTUM * v[pre];
        w += (eta * v[cur]);
    }

    // Prediction with w_MLE
    Matrix y = X * w;
    for (auto &row : y) {
        for (auto &e : row) {
            e = (e > 0.5) ? 1 : 0;
        }
    }

    vector<long double> gx1_1, gx1_2, gx2_1, gx2_2;
    int N = X.size() / 2;
    gx1_1.reserve(N); gx1_2.reserve(N); gx2_1.reserve(N); gx2_2.reserve(N);
    for (int i = 0; i != 2 * N; ++i) {
        switch (int(y[i][0])) {
            case 0: {
                gx1_1.emplace_back(X[i][1]);
                gx1_2.emplace_back(X[i][2]);
                break;
            }
            case 1: {
                gx2_1.emplace_back(X[i][1]);
                gx2_2.emplace_back(X[i][2]);
                break;
            }
        }
    }

    // Write the result to .csv
    FILE *fptr = fopen("result.csv", "a");
    if (fptr == nullptr) {
        perror("Failed to open result.csv");
        return -1;
    }
    
    switch (method) {
        case GradientDescent: fprintf(fptr, "\nGradient descent:\n"); break;
        case NewtonsMethod: fprintf(fptr, "\nNewton's method:\n"); break;
    }
    fprintf(fptr, "w:\n");
    for (int i = 0; i != w_n; ++i) {
        fprintf(fptr, "%LF,", w[i][0]);
    }    

    auto OutputCoord = [fptr](const vector<long double> &x) {
        for (int i = 0; i != x.size(); ++i) {
            fprintf(fptr, "%Lf,", x[i]);
        }
        fprintf(fptr, "\n");
    };
    fprintf(fptr, "\nX:");
    fprintf(fptr, "\ngx1:\n");
    OutputCoord(gx1_1);
    OutputCoord(gx1_2);
    fprintf(fptr, "gx2:\n");
    OutputCoord(gx2_1);
    OutputCoord(gx2_2);

    fclose(fptr);
    EvaluateAccuracy(t, y);

    return 0;
}


// Part 1. Logistic Regression
int LogisticRegression() {
    int N;
    int mx1, vx1, my1, vy1;
    int mx2, vx2, my2, vy2;

    printf("N = ");
    scanf("%d", &N);
    printf("mx1 = ");
    scanf("%d", &mx1);
    printf("my1 = ");
    scanf("%d", &my1);
    printf("mx2 = ");
    scanf("%d", &mx2);
    printf("my2 = ");
    scanf("%d", &my2);
    printf("vx1 = ");
    scanf("%d", &vx1);
    printf("vy1 = ");
    scanf("%d", &vy1);
    printf("vx2 = ");
    scanf("%d", &vx2);
    printf("vy2 = ");
    scanf("%d", &vy2);

    vector<pair<long double, long double>> D1(N), D2(N);
    long double sx1 = sqrt(vx1);
    long double sy1 = sqrt(vy1);
    long double sx2 = sqrt(vx2);
    long double sy2 = sqrt(vy2);
    for (int i = 0; i != N; ++i) {
        D1[i].first = GaussianDataGenerator(mx1, sx1);
        D1[i].second = GaussianDataGenerator(my1, sy1);
        D2[i].first = GaussianDataGenerator(mx2, sx2);
        D2[i].second = GaussianDataGenerator(my2, sy2);
    }

    Matrix X(2 * N, 3);
    Matrix t(2 * N, 1);
    for (int i = 0; i != N; ++i) {
        X[i][0] = 1;
        X[i][1] = D1[i].first;
        X[i][2] = D1[i].second;
        t[i][0] = 0;
    }
    for (int i = N, idx = 0; i != N * 2 && idx != N; ++i, ++idx) {
        X[i][0] = 1;
        X[i][1] = D2[idx].first;
        X[i][2] = D2[idx].second;
        t[i][0] = 1;
    }

    // Write to result.csv
    FILE *fptr = fopen("result.csv", "w");
    if (fptr == nullptr) {
        perror("Failed to open result.csv");
        return -1;
    }
 
    fprintf(fptr, "X:\nx1:\n");
    for (int i = 0; i != N; ++i) {
        fprintf(fptr, "%Lf,", X[i][1]);
    }
    fprintf(fptr, "\n");
    for (int i = 0; i != N; ++i) {
        fprintf(fptr, "%Lf,", X[i][2]);
    }

    fprintf(fptr, "\n");

    fprintf(fptr, "x2:\n");
    for (int i = N; i != N * 2; ++i) {
        fprintf(fptr, "%Lf,", X[i][1]);
    }
    fprintf(fptr, "\n");
    for (int i = N; i != N * 2; ++i) {
        fprintf(fptr, "%Lf,", X[i][2]);
    }

    fprintf(fptr, "\n");

    fclose(fptr);

    // Gradient Descent
    Descent(X, t, GradientDescent);
    Descent(X, t, NewtonsMethod);

    return 0;
}


// Part 2. EM Algorithm
const int IMAGE_HEIGHT = 28;
const int IMAGE_WIDTH = 28;
const int IMAGE_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH;
const int IMAGE_NUM = 60000;
const int BYTE = 1;
const int CLASS = 10;
const int MAX_ITER = 256;
const unsigned char THRESHOLD = 128;

template<typename T>
int argmax(const vector<T> &v) {
    int n = v.size();
    T max = numeric_limits<T>::min();
    int max_idx = -1;
    for (int i = 0; i != n; ++i) {
        if (v[i] > max) {
            max = v[i];
            max_idx = i;
        }
    }

    return max_idx;
}

template<typename T>
pair<int, int> argmax(const vector<vector<T>> &m) {
    int rows = m.size();
    int cols = m[0].size();
    T max = numeric_limits<T>::min();
    pair<int, int> max_idx = {-1, -1};
    for (int i = 0; i != rows; ++i) {
        for (int j = 0; j != cols; ++j) {
            if (m[i][j] > max) {
                max = m[i][j];
                max_idx = make_pair(i, j);
            }
        }
    }

    return max_idx;
}

int readImage(FILE *fptr, vector<bitset<IMAGE_SIZE>> &images) {
    const char *err_msg = "Failed to read from Image.\n";
    if (fptr == nullptr) {perror(err_msg); return -1;}
    if (fseek(fptr, 16, SEEK_SET) != 0) {perror("Fail to seek.\n"); return -1;}

    images.resize(IMAGE_NUM);
    for (int i = 0; i != IMAGE_NUM; ++i) {
        for (int pixel = 0; pixel != IMAGE_SIZE; ++pixel) {
            unsigned char tmp;
            if (fread(&tmp, sizeof(unsigned char), 1, fptr) != 1) {
                perror("Fail to get pixel.\n");
                return -1;
            }

            if (tmp < THRESHOLD) {images[i].reset(pixel);}
            else {images[i].set(pixel);}
        }
    }

    return 0;
}

int readLabel(FILE *fptr, vector<unsigned char> &labels) {
    const char *err_msg = "Failed to read from Label.\n";
    if (fptr == nullptr) {perror(err_msg); return -1;}
    if (fseek(fptr, 8, SEEK_SET) != 0) {perror(err_msg); return -1;}

    labels.resize(IMAGE_NUM);
    for (int i = 0; i != IMAGE_NUM; ++i) {
        if (fread(&labels[i], sizeof(unsigned char), 1, fptr) != 1) {perror(err_msg); return -1;}
    }

    return 0;
}

int printImage(const bitset<IMAGE_SIZE> &image) {
    for (int i = 0; i != IMAGE_SIZE; i += IMAGE_WIDTH) {
        for (int j = 0; j != IMAGE_WIDTH; ++j) {
            if (image[i + j]) {printf("* ");}
            else {printf(". ");}
        }
        printf("\n");
    }

    return 0;
}

vector<vector<long double>> E_step(const vector<bitset<IMAGE_SIZE>> &images,
                                   const vector<long double> &Lambda,
                                   const vector<vector<long double>> &Prob) {
    vector<vector<long double>> Z(IMAGE_NUM, vector<long double>(CLASS, 0.0));
    for (int i = 0; i != IMAGE_NUM; ++i) {
        long double marginal = 0;

        for (int j = 0; j != CLASS; ++j) {
            Z[i][j] = Lambda[j];
            long double &post = Z[i][j];

            for (int k = 0; k != IMAGE_SIZE; ++k) {
                if (images[i][k]) {post *= Prob[j][k];}
                else {post *= (1.0 - Prob[j][k]);}
            }
            marginal += post;
        }
        if (marginal == 0.0) {continue;}

        for (auto &e : Z[i]) {
            e /= marginal; 
        }
    }

    return Z;
}

int M_step(const vector<bitset<IMAGE_SIZE>> &images, const vector<vector<long double>> &Z,
           vector<long double> &Lambda, vector<vector<long double>> &Prob) {
    for (int i = 0; i != CLASS; ++i) {
        long double sum = 0.0;
        for (int j = 0; j != IMAGE_NUM; ++j) {
            sum += Z[j][i];
        }
        Lambda[i] = sum / IMAGE_NUM;

        for (int j = 0; j != IMAGE_SIZE; ++j) {
            long double tmp = 0.0;

            for (int k = 0; k != IMAGE_NUM; ++k) {
                if (images[k][j]) {
                    tmp += Z[k][i];
                }
            }
            (sum != 0.0) ? Prob[i][j] = tmp / sum : 0;
        }
    }

    return 0;
}

int printImagination(const vector<vector<long double>> &Prob, const vector<int> &mapping) {
    for (int i = 0; i != CLASS; ++i) {
        printf("class %d:\n", i);
        for (int row = 0; row != IMAGE_SIZE; row += IMAGE_WIDTH) {
            for (int col = 0; col != IMAGE_WIDTH; ++col) {
                if (Prob[mapping[i]][row + col] < 0.5) {printf(". ");}
                else {printf("* ");}
            }
            printf("\n");
        }       
        printf("\n");
    }

    return 0;
}

pair<vector<int>, vector<int>> assignLabel(const vector<bitset<IMAGE_SIZE>> &images,
                        const vector<unsigned char> &labels,
                        const vector<vector<long double>> &Z) {
    vector<int> mapping(CLASS);
    vector<int> pred(IMAGE_NUM);
    vector<vector<int>> count(CLASS, vector<int>(CLASS, 0));
    for (int i = 0; i != IMAGE_NUM; ++i) {
        auto &y = pred[i];
        y = argmax(Z[i]);
        ++count[labels[i]][y];
    }

    vector<int> mapping_inverse(CLASS);
    for (int i = 0; i != CLASS; ++i) {
        auto [label, class_idx] = argmax(count);
        mapping[label] = class_idx;
        mapping_inverse[class_idx] = label;
        for (int idx = 0; idx != CLASS; ++idx) {
            count[label][idx] = -1;
            count[idx][class_idx] = -1;
        }
    }

    for (auto &e : pred) {
        e = mapping_inverse[e];
    }
    
    return {mapping, pred};
}

int printConfusion(const vector<vector<int>> &Confusion, int num) {
    printf("\nConfusion Matrix %d:\n", num);
    printf("                Predict number %d Predict number not %d\n", num, num);
    printf("Is number %d        %-11d        %d\n", num, Confusion[0][0], Confusion[0][1]);
    printf("Isn't number %d     %-11d        %d\n", num, Confusion[1][0], Confusion[1][1]);
    printf("\nSensitivity (Successfully predict number %d)    : %f\n", num,
           (float)Confusion[0][0] / (Confusion[0][0] + Confusion[1][0]));
    printf("Specificity (Successfully predict not number %d): %f\n", num,
           (float)Confusion[1][1] / (Confusion[1][1] + Confusion[0][1]));
    printf("----------------------------------------------------------------\n");

    return 0;
}

int printResult(const vector<unsigned char> &labels, const vector<int> &pred, int iter) {
    vector<vector<vector<int>>> Confusion(10, vector<vector<int>>(2, vector<int>(2, 0)));
    int errors = IMAGE_NUM;
    for (int i = 0; i != IMAGE_NUM; ++i) {
        for (int j = 0; j != CLASS; ++j) {
            if (labels[i] == j) {
                if (pred[i] == j) {
                    ++Confusion[j][0][0];
                    --errors;
                }
                else {++Confusion[j][0][1];}
            }
            else {
                (pred[i] == j) ? ++Confusion[j][1][0] : ++Confusion[j][1][1];
            }
        }
    }

    for (int i = 0; i != CLASS; ++i) {
        printConfusion(Confusion[i], i);
    }
    printf("Total iteration to converge: %d\n", iter);
    printf("Total error rate: %f\n", (float)errors / IMAGE_NUM);

    return 0;
} 

int EM_Algorithm() {
    // Read data from MNIST
    vector<bitset<IMAGE_SIZE>> images;
    vector<unsigned char> labels;
    // Read training image
    const char *train_image = "train-images.idx3-ubyte"; 
    FILE *fptr = fopen(train_image, "rb");
    if (readImage(fptr, images) != 0) {
        fprintf(stderr, "Fail to read from %s.\n", train_image);
        return -1;
    }
    fclose(fptr);

    // Read training label
    const char *train_label = "train-labels.idx1-ubyte";
    fptr = fopen(train_label, "rb");
    if (readLabel(fptr, labels) != 0) {
        fprintf(stderr, "Faile to read from %s.\n", train_label);
        return -1;
    }
    fclose(fptr);
    fptr = nullptr;

    // Initialization
    vector<long double> Lambda;
    vector<vector<vector<long double>>> Prob(2, vector<vector<long double>>(CLASS,
                                                vector<long double>(IMAGE_SIZE, 0.0)));

    // Functions used in every iteration
    int cur_iter = 0;
    auto zero = [&]() {
        for (const auto &e : Lambda) {
            if (e == 0) {return true;}
        }
        return false;
    };

    int cur, prev;
    auto reset = [&]() {
        cur = cur_iter & 1, prev = (~cur) & 1;
        Lambda = vector<long double>(10, 0.1);
        for (int i = 0; i != CLASS; ++i) {
            for (int j = 0; j != IMAGE_SIZE; ++j) {
                Prob[cur][i][j] = RandomProbability();
                Prob[prev][i][j] = RandomProbability();
            }
        }

        return;
    };

    long double delta;
    auto calculate_delta = [&]() {
        delta = 0;
        long double base = 0;
        for (int i = 0; i != CLASS; ++i) {
            for (int j = 0; j != IMAGE_SIZE; ++j) {
                delta += abs(Prob[cur][i][j] - Prob[prev][i][j]);
                base += abs(Prob[cur][i][j]);
            }
        }
        if (delta < base * 1e-3 && cur_iter > 16) {return true;}
        return false;
    };

    // Start EM Algorithm
    int iter = 0;
    vector<vector<long double>> Z(IMAGE_NUM, vector<long double>(CLASS, 0.0));
    vector<int> unmap(CLASS);
    for (int i = 0; i != CLASS; ++i) {unmap[i] = i;}
    reset();
    bool isConverged;
    do {
        cur = cur_iter & 1; prev = (~cur) & 1;
        Z = E_step(images, Lambda, Prob[prev]);
        M_step(images, Z, Lambda, Prob[cur]);
        isConverged = calculate_delta();

        printImagination(Prob[cur], unmap);
        printf("No. of Iteration: %d, Difference: %.10Lf\n", iter + 1, delta);
        printf("----------------------------------------------------------------\n");
        ++iter;

        if (zero()) {
            reset();
            cur_iter = 0;
            continue;
        }
        else {
            ++cur_iter;
        }

    } while (!isConverged && iter < MAX_ITER);

    printf("----------------------------------------------------------------\n\n");
    auto [mapping, pred] = assignLabel(images, labels, Z);
    printImagination(Prob[cur], mapping);
    printResult(labels, pred, iter);
    
    return 0;
}

int main(int argc, char *argv[]) {
    auto PrintUsage = []() {
        printf("Usage: ./program <part>\n");
        printf("<part> = \n1: run part 1.\n2: run part 2.\n3: run part 1. and 2.\n");
    };

    if (argc != 2 || !isdigit(*argv[1])) {
        PrintUsage();
        return 0;
    }

    switch (stoi(argv[1])) {
        case 1: LogisticRegression(); break;
        case 3: LogisticRegression();
        case 2: EM_Algorithm(); break;
        default: PrintUsage(); break;
    }
 
    return 0;
}