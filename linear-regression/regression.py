import matplotlib.pyplot as plt

def Deep_Copy(lst):
    return [Deep_Copy(item) if isinstance(item, list) else item for item in lst]


class Matrix:
    def __init__(self, matrix, n=-1):
        if isinstance(matrix, int):
            self.matrix = [[int(i == j) for j in range(n)] for i in range(n)]
            self.rows = n
            self.cols = n
        elif isinstance(matrix, list):
            self.matrix = Deep_Copy(matrix)
            self.rows = len(matrix)
            self.cols = len(matrix[0]) if self.rows > 0 else 0
        else: raise ValueError("Constructing error.")
        
    def __str__(self):
        ret = ",\n ".join(map(str, self.matrix))
        return "[" + ret + "]"

    def __getitem__(self, idx):
        return self.matrix[idx]
    
    def __add__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Addition error.")

        ret = []
        for left_row, right_row in zip(self.matrix, other.matrix):
            ret.append([left_row[j] + right_row[j] for j in range(self.cols)])
        return Matrix(ret)

    def __neg__(self):
        return Matrix([[-entry for entry in row] for row in self.matrix]) 

    def __sub__(self, other):
        return self + -other   

    def __matmul__(self, other):
        if self.cols != other.rows:
            raise ValueError("Multiplication error.")
        
        ret = []
        for i, self_row in enumerate(self.matrix):
            ret.append([])
            for k in range(other.cols):
                ret[i].append(sum([self_row[j] * other.matrix[j][k] for j in range(self.cols)]))

        return Matrix(ret)

    def __mul__(self, other):
        assert isinstance(other, int) or isinstance(other, float)
        return Matrix([[entry * other for entry in row] for row in self.matrix])
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __pow__(self, other):
        if self.rows != self.cols:
            raise ValueError("Power error.")
        n = self.rows

        if other < 0:
            A = self.Inverse()            
            other *= -1
        else:
            A = Matrix(self.matrix)

        if other == 1:
            ret = A
        else:
            ret = Matrix(1, self.rows)
            for _ in range(other):
                ret @= A
        
        return ret

    def Switch(self, row1, row2):
        if row1 >= self.rows or row2 >= self.rows:
            raise ValueError("Row out of range.")
        elif row1 == row2:
            return

        tmp = self.matrix[row1]
        self.matrix[row1] = self.matrix[row2]
        self.matrix[row2] = tmp

        return

    def LU_Decomposition(self):
        if self.rows != self.cols:
            raise ValueError("Cannot decomposition")
        
        n = self.rows
        L, U = Matrix(1, self.rows), Matrix(self.matrix)
        for i in range(n):
            for row in range(i, n):
                if U.matrix[row][i]: break
            U.Switch(i, row)
            pivot = U.matrix[i][i]

            for row in range(i + 1, n):
                if not pivot: break

                elimination = L.matrix[row][i] = U.matrix[row][i] / pivot
                for j in range(i, n):
                    U.matrix[row][j] -= elimination * U.matrix[i][j]

        return L, U
    
    def Solving_Linear_System(self, B):
        L, U = self.LU_Decomposition()
        n = B.cols
        Y = Matrix([[0 for _ in B_row] for B_row in B.matrix]) 
        for row, L_row in enumerate(L.matrix):
            for j in range(n):
                tmp = B.matrix[row][j] - sum(L_row[i] * Y.matrix[i][j] for i in range(row))
                Y.matrix[row][j] = tmp / L_row[row]
        
        X = Matrix([[0 for _ in B_row] for B_row in B.matrix])
        for index, U_row in enumerate(reversed(U.matrix)):
            row = n - index - 1
            for j in range(n - 1, -1, -1):
                tmp = sum(U_row[i] * X.matrix[i][j] for i in range(n - 1, row, -1))
                X.matrix[row][j] = (Y.matrix[row][j] - tmp) / U_row[row]

        return X 

    def Inverse(self):
        if self.rows != self.cols:
            raise ValueError("Cannot inverse.")
       
        n = self.rows
        # Solving inverse by LU decomposition
        return self.Solving_Linear_System(Matrix(1, n))

    def Transpose(self):
        return Matrix([[self.matrix[j][i] for j in range(self.rows)] for i in range(self.cols)])


def F(x, w):
    return sum([w_i[0] * (x ** e) for e, w_i in enumerate(w.matrix)])

def Show_Result(x_vector, y_vector, w):
    # Print fitting line
    n = len(w.matrix) - 1
    f_x = [f"{w_i[0]}X^{n - i}" for i, w_i in enumerate(w.matrix[:0:-1])]
    f_x.append(str(w.matrix[0][0]))
    print("Fitting line: " + " + ".join(f_x))

    # Print error
    n += 1
    A = Matrix([[x ** i for i in range(n)] for x in x_vector])
    E = A @ w - Matrix([[y] for y in y_vector])
    print("Total error: " + str((E.Transpose() @ E).matrix[0][0]))

    # Visualization
    plt.scatter(x_vector, y_vector)
    plt.plot(x_vector, [F(x, w) for x in x_vector])
    plt.show()

    return

def LSE_L2_norm(x_vector, y_vector, N, LAMBDA):
    A = Matrix([[x ** e for e in range(N)] for x in x_vector])
    I = Matrix(1, N)
    w = (A.Transpose() @ A + LAMBDA * I) ** (-1) @ A.Transpose() @ Matrix([[y] for y in y_vector])

    print("LSE:")
    Show_Result(x_vector, y_vector, w)
    return

def L1_norm(vector):
    return sum([abs(e[0]) for e in vector.matrix])

def L1_d(ATA, ATy, w, LAMBDA):
    lT = Matrix([[LAMBDA] if w_i[0] > 0 else ([0] if w_i[0] == 0 else [-LAMBDA]) for w_i in w.matrix])
    return 2 * ATA @ w - 2 * ATy + lT

def Steepest_Descent(x_vector, y_vector, N, LAMBDA, lr=1e-5, tol=1e-5, max_iter=100):
    A = Matrix([[x ** e for e in range(N)] for x in x_vector])
    ATA = A.Transpose() @ A
    ATy = A.Transpose() @ Matrix([[y_i] for y_i in y_vector])
    w = Matrix([[0] for _ in range(N)])

    # Steepest Descent
    for _ in range(max_iter):
        d = L1_d(ATA, ATy, w, LAMBDA)
        if L1_norm(d) < tol:
            break

        w = w - lr * d

    print("Steepest descent:")
    Show_Result(x_vector, y_vector, w)
    return

def Newtons_Method(x_vector, y_vector, N, LAMBDA, lr=1, tol=1e-5, max_iter=100):
    A = Matrix([[x ** e for e in range(N)] for x in x_vector])
    H = 2 * A.Transpose() @ A
    w = Matrix([[0] for _ in range(N)])
    ym = Matrix([[y_i] for y_i in y_vector])

    # Newton's Method
    for _ in range(max_iter):
        g = H @ w - 2 * A.Transpose() @ ym
        d = H ** (-1) @ g
        if L1_norm(d) < tol:
            break
        w = w - lr * d
    
    print("Newton's Method:")
    Show_Result(x_vector, y_vector, w)
    return


if __name__ == "__main__":
    # Reading file
    x_vector, y_vector = list(), list()
    with open("testfile.txt", "r") as file:
        for line in file:
            pair = line.strip().split(',')
            if len(pair) == 2:
                x_vector.append(float(pair[0]))
                y_vector.append(float(pair[1]))
            else: raise ValueError("Fail to load testfile.txt")

    # Input parameters
    N = int(input("n = "))
    LAMBDA = int(input("lambda = "))

    LSE_L2_norm(x_vector, y_vector, N, LAMBDA)
    print("\n\n")
    Steepest_Descent(x_vector, y_vector, N, LAMBDA)
    print("\n\n")
    Newtons_Method(x_vector, y_vector, N, LAMBDA)