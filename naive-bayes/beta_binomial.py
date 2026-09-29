fac = [1]
def factorial(x):
    n = len(fac) - 1
    while (x > n):
        fac.append(fac[n] * (n + 1))
        n += 1
    
    return fac[x]

def beta_conjugation(a, b, N, m):
    N += m
    a_post = a + m
    b_post = b + N - m

    # p = a_post / (a_post + b_post)
    p = m / N
    likelihood = factorial(N) / factorial(m) / factorial(N - m) * (p ** m) * ((1 - p) ** (N - m))

    print(f"Likelihood: {likelihood}")
    print(f"Beta prior:     a = {a} b = {b}")
    print(f"Beta posterior: a = {a_post} b = {b_post}")
    return a_post, b_post


if __name__ == "__main__":
    print("Enter beta prior:")
    a = int(input("a = "))
    b = int(input("b = "))

    with open('testfile.txt', 'r') as file:
        for i, line in enumerate(file):
            print(f"case {i + 1}: " + line)
            a, b = beta_conjugation(a, b, line.count('0'), line.count('1'))
