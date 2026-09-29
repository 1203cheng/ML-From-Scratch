#include "Matrix.h"
#include <vector>
#include <cassert>
#include <cstdio>
using namespace std;


// Constructor
Matrix::Matrix(const Matrix &other) : n(other.n), m(other.m), data(other.data) {}
Matrix::Matrix(Matrix &&other) noexcept : n(other.n), m(other.m), data(std::move(other.data)) {}
Matrix::Matrix(const vector<long double> &vec) : n(vec.size()), m(1),
                                                 data(n, vector<long double>(m, 0)) {
    for (int i = 0; i != n; ++i) {
        data[i][0] = vec[i];
    }
    
    return;
}

Matrix::Matrix(const vector<vector<long double>> &mat) : n(mat.size()), m(mat[0].size()),
                                                         data(n, vector<long double>(m, 0)) {
    for (int i = 0; i != n; ++i) {
        data[i] = mat[i];
    }

    return;
}

Matrix::Matrix(int n, int m, long double diag) : n(n), m(m),
                                                 data(n, vector<long double>(m, 0)) {
    if (diag == 0) {return;}
    for (int i = 0; i < n && i < m; ++i) {
        data[i][i] = diag;
    }

    return;
}

// Copy and Move Assignment
Matrix& Matrix::operator =(const Matrix &other) {
    if (this != &other) {
        n = other.n;
        m = other.m;
        data.resize(n);
        for (int i = 0; i != n; ++i) {
            data[i] = other.data[i];
        }
    }

    return *this;
}

Matrix& Matrix::operator =(Matrix &&other) noexcept {
    if (this != &other) {
        n = other.n;
        m = other.m;
        data = std::move(other.data);
    } 

    return *this;
}

// Overload subscript and iterator
vector<long double>& Matrix::operator [](const int &i) {
    assert(i < n);
    return data[i];
}

const vector<long double>& Matrix::operator [](const int &i) const {
    assert(i < n);
    return data[i];
}

vector<vector<long double>>::iterator Matrix::begin() {return data.begin();}
vector<vector<long double>>::iterator Matrix::end() {return data.end();}
vector<vector<long double>>::const_iterator Matrix::begin() const {return data.begin();}
vector<vector<long double>>::const_iterator Matrix::end() const {return data.end();}

// Overload basic arithmetic operator
const Matrix Matrix::operator +(const Matrix &other) const {
    assert(n == other.n && m == other.m);

    Matrix ret(n, m);
    for (int i = 0; i != n; ++i) {
        for (int j = 0; j != m; ++j) {
            ret[i][j] = data[i][j] + other.data[i][j];
        }
    }
    
    return ret;
}

const Matrix Matrix::operator -() const {
    Matrix ret(data);
    for (vector<long double> &row : ret) {
        for (long double &col : row) {
            col = -col;
        }
    }

    return ret;
}

const Matrix Matrix::operator -(const Matrix &other) const {
    return *this + (-other);
}
const Matrix Matrix::operator *(const Matrix &other) const {
    assert(m == other.n);

    Matrix ret(n, other.m);
    for (int i = 0; i != n; ++i) {
        for (int j = 0; j != other.m; ++j) {
            for (int k = 0; k != m; ++k) {
                ret[i][j] += data[i][k] * other.data[k][j];
            }
        }
    }

    return ret;
}

const Matrix Matrix::operator *(const long double &s) const {
    Matrix ret(*this);
    for (auto &row : ret) {
        for (auto &col : row) {
            col *= s;
        }
    }

    return ret;
}

Matrix& Matrix::operator +=(const Matrix &other) {
    assert(n == other.n && m == other.m);
    for (int i = 0; i != n; ++i) {
        for (int j = 0; j != m; ++j) {
            data[i][j] += other[i][j];
        }
    }

    return *this;
}
Matrix& Matrix::operator -=(const Matrix &other) {
    *this += -other;
    return *this;
}
Matrix& Matrix::operator *=(const long double &s) {
    for (auto &row : *this) {
        for (auto &col : row) {
            col *= s;
        }
    }

    return *this;
}

// Overload compare operator
const bool Matrix::operator ==(const Matrix &other) const {
    if (n != other.n || m != other.m) {return false;}
    for (int i = 0; i != n; ++i) {
        for (int j = 0; j != m; ++j) {
            if (data[i][j] != other.data[i][j]) {return false;}
        }
    }

    return true;
}

const bool Matrix::operator !=(const Matrix &other) const {
    return !(*this == other);
}

// Capacity and Modifier
const size_t Matrix::size() const {return n;}
int Matrix::switch_rows(int idx1, int idx2) {
    if (idx1 >= n || idx2 >= n) {
        fprintf(stderr, "Switching row %d and %d out of range %d", idx1, idx2, n);
        return -1;
    }
    else if (idx1 == idx2) {return 0;}

    for (int j = 0; j != m; ++j) {
        long double tmp = data[idx1][j];
        data[idx1][j] = data[idx2][j];
        data[idx2][j] = tmp; 
    }
    
    return 0;
}


// Non-member function
// Overload basic arithmetic operator
const Matrix operator *(const long double &s, const Matrix &other) {
    return other * s;
}

// Complex arithmetic for Matrix
const Matrix Transpose(const Matrix &mat) {
    int n = mat.size();
    int m = mat[0].size();
    Matrix ret(m, n);
    
    for (int i = 0; i != n; ++i) {
        for (int j = 0; j != m; ++j) {
            ret[j][i] = mat[i][j];
        }
    }

    return ret;
}

const pair<Matrix, Matrix> LU_Decompose(const Matrix &mat) {
    int n = mat.size();
    // test
    // printf("n = %d\n", n); //
    assert(n == mat[0].size());

    Matrix L(n, n, 1), U(mat);
    for (int i = 0; i != n; ++i) {
        int row;
        // test
        // printf("i = %d\n\trow = ", i); //
        for (row = i; row != n; ++row) {
            // test
            // printf("%d, ", row); //
            if (U[row][i]) {break;}
        }
        // test
        // printf("\n");
        assert(row != n);

        U.switch_rows(i, row);
        long double pivot = U[i][i];
        if (pivot == 0) {continue;}

        for (row = i + 1; row != n; ++row) {
            // test
            // printf("\trow = %d\n\t\tj = ", row); //
            long double elim = L[row][i] = U[row][i] / pivot;
            for (int j = i; j != n; ++j) {
                // test
                // printf("%d, ", j); //
                U[row][j] -= (elim * U[i][j]);
            }
            //test
            // printf("\n"); //
        }
    }

    //test
    // printf("LU complete\n"); //
    return {L, U};
}

const Matrix SolvingLinearSystem(const Matrix &A, const Matrix &B) { // AX = B, X = ?
    int n = A.size();
    assert(n == B.size());

    int p = B[0].size();
    auto [L, U] = LU_Decompose(A);
    Matrix Y(n, p), X(n, p);

    long double tmp, sum;
    for (int row = 0; row != L.size(); ++row) {
        for (int j = 0; j != p; ++j) {
            sum = 0;
            for (int i = 0; i != row; ++i) {
                // test
                // printf("%d, %d, %d\n", row, j, i); //
                sum += (L[row][i] * Y[i][j]);
            }
            tmp = B[row][j] - sum;
            Y[row][j] = tmp / L[row][row];
        }
    }
    // test
    // printf("\n"); //

    for (int row = p - 1; row >= 0; --row) {
        for (int j = p - 1; j >= 0; --j) {
            sum = 0;
            for (int i = p - 1; i != row; --i) {
                //test
                // printf("%d, %d, %d\n", row, j, i); //
                sum += (U[row][i] * X[i][j]);
            }
            X[row][j] = (Y[row][j] - sum) / U[row][row];
        }
    }

    return X;
}

const Matrix Inverse(const Matrix &mat) {
    int n = mat.size();
    assert(n == mat[0].size());

    return SolvingLinearSystem(mat, Matrix(n, n, 1));
}