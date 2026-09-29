import matplotlib.pyplot as plt
import csv
from typing import TextIO

def FindVariable(file: TextIO,variable: str) -> float:
    for line in file:
        if line.startswith(variable):
            return float((line.split(','))[1])

    raise ValueError(f"{variable} not found in the file.")

def FindLine(file: TextIO, string: str) -> None:
    for line in file:
        if line.strip() == string: return

    raise ValueError(f"\"{string}\" not found in the file.")

def ParseMu(file: TextIO, n: int) -> list:
    ret = []
    reader = csv.reader(file)

    for idx, row in enumerate(reader):
        ret.append(float(row[0]))
        if idx == n - 1: break

    return ret

def ParseMatrix(file: TextIO, n: int) -> list:
    matrix = []
    reader = csv.reader(file)

    for idx, row in enumerate(reader):
        matrix.append([float(entry) for entry in row if entry])
        if idx == n - 1: break

    return matrix

def GetY(X: list, w: list) -> list:
    assert(len(X[0]) == len(w))

    ret = []
    for x in X:
        ret.append(sum([e * w_i for (e, w_i) in zip(x, w)]))

    return ret

def GetDif(X: list, Lambda_i: list) -> float:
    assert(len(X) == len(Lambda_i) == len(Lambda_i[0]))
    tmp = [0] * len(X)
    for i in range(len(X)):
        for idx, x in enumerate(X):
            tmp[i] += (x * Lambda_i[idx][i])

    return sum([e * x for e, x in zip(tmp, X)])
 
def SubplotBackGround(idx: int,
                      title: str,
                      X: list,
                      w: list,
                      a: float,
                      Lambda_i: list = None) -> None:
    plt.subplot(idx)
    plt.title(title)
    plt.xlim(-2.0, 2.0)
    plt.ylim(-20.0, 20.0)

    x_curve = [x[1] for x in X]
    y_curve = GetY(X, w)
    plt.plot(x_curve, y_curve, 'k')


    y_curve_upper, y_curve_lower = [y for y in y_curve], [y for y in y_curve]
    if Lambda_i == None:
        y_curve_upper = [y + a for y in y_curve_upper]
        y_curve_lower = [y - a for y in y_curve_lower]
    else:
        for idx, x in enumerate(X):
            dif = GetDif(x, Lambda_i) + a
            y_curve_upper[idx] += dif
            y_curve_lower[idx] -= dif

    plt.plot(x_curve, y_curve_upper, 'r')
    plt.plot(x_curve, y_curve_lower, 'r')

    return

   

def PlotVariance(X: list, XT: list, Lambda_i: list) -> None:
    x_curve = [x[1] for x in X]


if __name__ == "__main__":
    # Initialize variables
    n, a, b = 0, 0.0, 0.0
    w = []
    mu_10, mu_50, mu = [], [], []
    Lambda_i_10, Lambda_i_50, Lambda_i = [], [], []
    X, XT = [], []
    x, y = [], []

    # Read the file
    with open('result.csv', 'r') as file:
        n = int(FindVariable(file, 'n'))
        a = FindVariable(file, 'a')
        b = FindVariable(file, 'b')

        FindLine(file, "w:")
        w = ParseMatrix(file, 1)
        w = [e for e in w[0]]

        FindLine(file, "Mu 10:")
        mu_10 = ParseMu(file, n)
        FindLine(file, "Mu 50:")
        mu_50 = ParseMu(file, n)
        FindLine(file, "Mu:")
        mu = ParseMu(file, n)

        FindLine(file, "Lambda_i 10:")
        Lambda_i_10 = ParseMatrix(file, n)
        FindLine(file, "Lambda_i 50:")
        Lambda_i_50 = ParseMatrix(file, n)
        FindLine(file, "Lambda_i:")
        Lambda_i = ParseMatrix(file, n)

        FindLine(file, "Design matrix X:")
        reader = csv.reader(file)
        for idx, row in enumerate(reader):
            X.append([float(entry) for entry in row if entry])
            if idx == 29: break

        FindLine(file, "Data:")
        reader = csv.reader(file)
        for row in reader:
            x.append(float(row[0]))
            y.append(float(row[1]))

    # print(X)
    # print(XT)

    # Plotting
    SubplotBackGround(221, "Ground truth", X, w, a)
    SubplotBackGround(222, "Predict result", X, mu, a, Lambda_i)
    plt.scatter(x, y, 10)
    SubplotBackGround(223, "After 10 incomes", X, mu_10, a, Lambda_i_10)
    plt.scatter(x[:10], y[:10], 10)
    SubplotBackGround(224, "After 50 incomes", X, mu_50, a, Lambda_i_50)
    plt.scatter(x[:50], y[:50], 10)

    plt.tight_layout
    plt.show()
   