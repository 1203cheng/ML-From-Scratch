#ifndef __MATRIX_H__
#define __MATRIX_H__
#include <vector>
#include <utility>

class Matrix {
    int n, m;
    std::vector<std::vector<long double>> data;
public:
    // Constructor
    Matrix(const Matrix &other);
    Matrix(Matrix &&other) noexcept;
    Matrix(const std::vector<long double> &vec);
    Matrix(const std::vector<std::vector<long double>> &mat);
    Matrix(int n = 0, int m = 0, long double diag = 0.0);

    // Copy and Move Assignment
    Matrix& operator =(const Matrix &other);
    Matrix& operator =(Matrix &&other) noexcept;

    // Overload subscript and iterator
    std::vector<long double>& operator [](const int &i);
    const std::vector<long double>& operator [](const int &i) const;
    std::vector<std::vector<long double>>::iterator begin();
    std::vector<std::vector<long double>>::iterator end();
    std::vector<std::vector<long double>>::const_iterator begin() const;
    std::vector<std::vector<long double>>::const_iterator end() const;

    // Overload basic arithmetic operator
    const Matrix operator +(const Matrix &other) const;
    const Matrix operator -() const;
    const Matrix operator -(const Matrix &other) const;
    const Matrix operator *(const Matrix &other) const;
    const Matrix operator *(const long double &s) const;
    Matrix& operator +=(const Matrix &other);
    Matrix& operator -=(const Matrix &other);
    Matrix& operator *=(const long double &s);

    // Overload compare operator
    const bool operator ==(const Matrix &other) const;
    const bool operator !=(const Matrix &other) const;

    // Capacity and Modifier
    const size_t size() const;
    int switch_rows(int idx1, int idx2);
};
// Overload basic arithmetic operator
const Matrix operator *(const long double &s, const Matrix &other);

// Complex arithmetic for Matrix
const Matrix Transpose(const Matrix &mat);
const std::pair<Matrix, Matrix> LU_Decompose(const Matrix &mat);
const Matrix SolvingLinearSystem(const Matrix &A, const Matrix &B); // AX = B, X = ?
const Matrix Inverse(const Matrix &mat);

#endif