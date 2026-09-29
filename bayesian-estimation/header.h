#ifndef __HEADER_H__
#define __HEADER_H__
#include <vector>
#include <utility>
#include <cstring>
#include "Matrix.h"

// Part 1.
long double GaussianDataGenerator(const long double &mu, const long double &sigma); 
std::pair<long double, long double> GaussianLinearData(const int &n, long double &sigma,
                                                       const Matrix &W); 

// Part 2.
class Data {
    int n;
    long double mean;
    long double M2;

public:
    Data(int n = 0, long double mean = 0.0, long double M2 = 0.0);
    void AddData(const long double &v);
    int get_n(); 
    long double get_mean(); 
    long double get_variance(); 
    long double get_sd(); 
};

// Part 3.
Matrix DesignMatrix(const int &n, const long double &x); 
void PrintPosterior(const Matrix &Mu, const Matrix &Lambda);
int SaveMatrix(FILE *fptr, const std::string &title, const Matrix &mat);
# endif