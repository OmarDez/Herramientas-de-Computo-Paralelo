from matplotlib import pyplot as plt
from numpy import *


class ecuacionCalor():
    # Condiciones iniciales
    def __init__(self, L):
        phi = []
        for i in arange(0, L/2, 0.01):
            phi.append(i)
        for i in arange(L / 2, L, 0.01):
            phi.append(L - i)
        self.phi = phi

    def b_n1(self, n, L):
        b = []
        for i in range(1, n):
            alpha = (pi*i)/2
            e_1 = 2 * sin(alpha)/(pi * i**2)
            e_2 = cos(alpha)/i
            s = e_1 - e_2
            b.append(s)
        return b

    def b_n2(self, n, L):
        b = []
        for i in range(1, n):
            e_1 = -2*sin(pi*i) + 2*sin((pi*i)/2) + pi*i*cos((pi*i)/2)
            e_2 = 2*i**2
            s = e_1/e_2
            b.append((2/L) * s)
        return b

    def serieFourier(self,B, k, t, L, limite_inferior, limite_superior):
        sumatoria = zeros(len(arange(limite_inferior, limite_superior, 0.01)))
        for n in range(len(B)):
            i = n + 1
            alpha = (i*pi)/L
            e = exp((-i**2*pi**2*k*t)/(L**2))
            s = array([sin(alpha * x) for x in arange(limite_inferior, limite_superior, 0.01)])
            sumatoria += B[n]*s*e
        return sumatoria

if __name__ == '__main__':
    k = 1
    L = pi
    n = 1000
    plt.xlabel('Posición de material')
    plt.ylabel('Temperatura')
    for t in range(4):
        y = ecuacionCalor(L)
        B1 = y.b_n1(n, L)
        B2 = y.b_n2(n, L)
        figura1 = y.serieFourier(B1, k, t, L, 0, L/2)
        figura2 = y.serieFourier(B2, k, t, L, L/2, L)
        figura = concatenate((figura1, figura2))
        plt.plot(y.phi)
        plt.plot(figura, label = 't = {}'.format(t))
    plt.legend()
    plt.show()
