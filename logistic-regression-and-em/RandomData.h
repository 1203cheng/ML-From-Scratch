#pragma once
#include <utility>
#include "Matrix.h"

long double RandomProbability();
// Notice that the argument is "sigma" not "variance"
long double GaussianDataGenerator(const long double &mu, const long double &sigma); 
std::pair<long double, long double> GaussianLinearData(const int &n, long double &sigma,
                                                       const Matrix &W); 

