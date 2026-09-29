import matplotlib.pyplot as plt
import csv
from typing import TextIO

def FindLine(file: TextIO, string: str) -> int:
    for line in file:
        if line.strip() == string: return 0

    return -1

def ParseVector(file: TextIO) -> list:
    vector = [float(entry) for entry in file.readline().strip().split(',') if entry]
    return vector

def ParseCoord(file: TextIO) -> tuple[list, list]:
    ret1 = ParseVector(file)
    ret2 = ParseVector(file)
    return ret1, ret2

def getResult():
    FindLine(file, "w:")
    w = ParseVector(file)
    FindLine(file, "X:")
    FindLine(file, "gx1:")
    gx1_1, gx1_2 = ParseCoord(file)
    FindLine(file, "gx2:")
    gx2_1, gx2_2 = ParseCoord(file)

    FindLine(file, "TP:")
    TP = int(file.readline().split(',')[0])
    FindLine(file, "FP:")
    FP = int(file.readline().split(',')[0])
    FindLine(file, "TN:")
    TN = int(file.readline().split(',')[0])
    FindLine(file, "FN:")
    FN = int(file.readline().split(',')[0])

    return w, gx1_1, gx1_2, gx2_1, gx2_2, TP, FP, TN, FN

def PrintResult(w: list, TP: int, FP: int, TN: int, FN: int) -> None:
    print("w:")
    for w_i in w:
        print(f"  {w_i}")
    print("\nConfusion Matrix:")
    print("             Predict cluster 1 Predict cluster 2")
    print(f"Is cluster 1        {TP:^17}        {FN}")
    print(f"Is cluster 2        {FP:^17}        {TN}")
    print(f"\nSensitivity (Successfully predict cluster 1): {TP / (TP + FP)}")
    print(f"Specificity (Successfully predict cluster 2): {TN / (TN + FN)}")

    return None


if __name__ == "__main__":
    x1_1, x1_2, x2_1, x2_2 = [], [], [], []
    gx1_1, gx1_2, gx2_1, gx2_2 = [], [], [], []
    w = []
    TP, FP, TN, FN = 0, 0, 0, 0

    # Read the file
    Newton = True
    with open ('result.csv', 'r') as file:
        FindLine(file, "X:")
        FindLine(file, "x1:")
        x1_1, x1_2 = ParseCoord(file)
  
        FindLine(file, "x2:")
        x2_1, x2_2 = ParseCoord(file)

        plt.subplot(131)
        plt.title("Ground truth")
        plt.scatter(x1_1, x1_2, c='r')
        plt.scatter(x2_1, x2_2, c='b') 

        # Gradient Descent
        w, gx1_1, gx1_2, gx2_1, gx2_2, TP, FP, TN, FN = getResult()

        print("Gradient descent:\n")
        PrintResult(w, TP, FP, TN, FN)

        plt.subplot(132)
        plt.title("Gradient descent")
        plt.scatter(gx1_1, gx1_2, c='r')
        plt.scatter(gx2_1, gx2_2, c='b')

        # Newton's Method
        w, gx1_1, gx1_2, gx2_1, gx2_2, TP, FP, TN, FN = getResult()
        
        print('-' * 64)
        print("Newton's method:\n")
        PrintResult(w, TP, FP, TN, FN)
        
        plt.subplot(133)
        plt.title("Newton's method")
        plt.scatter(gx1_1, gx1_2, c='r')
        plt.scatter(gx2_1, gx2_2, c='b')
        # if FindLine(file, "Newton's method:") != 0:
            # Newton = False
    plt.show()