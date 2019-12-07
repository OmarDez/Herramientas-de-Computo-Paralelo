from numpy import *
from matplotlib import pyplot as plt

#f(x) = 0, -3 <= x < 0
#f(x) = 2 + x, 0 <= x <= 3

def a_n(n):
    a = []
    for i in range(1, n):
        sen = ((-1)**i)-1
        a.append((3*sen)/(i**2 * pi**2))
    return a

def b_n(n):
    b = []
    for i in range(1, n):
        c = (-1) ** i
        b.append((2 - 5 * c) / (i * pi))
    return b

def serie(a_0, A, B, L):
    f = zeros(len(arange(0, L, 0.01)))

    for i in range(len(A)):
        alpha = ((i+1)*pi)/L
        c = array([cos(alpha * x) for x in arange(0, L, 0.01)])
        s = array([sin(alpha * x) for x in arange(0, L, 0.01)])

        e_1 = A[i] * c
        e_2 = B[i] * s
        f += e_1 + e_2

    return a_0 + f

if __name__ == '__main__':
    a_0 = 7/2
    L = 3
    n = 100000
    A = a_n(n)
    B = b_n(n)
    figura1 = serie(a_0, A, B, L)
    plt.plot(figura)
    plt.show()