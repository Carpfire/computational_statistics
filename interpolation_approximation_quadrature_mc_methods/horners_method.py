from matplotlib import interactive
from numpy.polynomial.chebyshev import Chebyshev
from scipy.integrate import quad
import numpy as np 
import matplotlib.pyplot as plt 


def horners(coef, x, r, t, s):
#Polynomial evaluation through horner method
#Input: Polynomial coefficients, evaluation point 
#Output: Approximation of function at evaluation point 
    deg = coef.shape[0]
    f_k2 = coef[-1]
    f_k1 = coef[-2] + f_k2*(r*x-s)
    for k in range(deg-2, 0, -1):
        f_k = coef[k] + f_k1*(r*x-s) - f_k2*t
        f_k2 = f_k1
        f_k1 = f_k 
    p_0 = coef[0] + f_k1*x - f_k2 
    return p_0

def main():
    cheb_coef = Chebyshev.interpolate(np.exp, 5, [-1,1]).coef
    domain = np.linspace(-1,1)
    error= abs(horners(cheb_coef, 0, 2, 1, 0) - np.exp(0))
    approx = horners(cheb_coef, domain, 2, 1, 0)
    true = np.exp(domain)
    squared_error = lambda x: (horners(cheb_coef, x, 2, 1, 0) - np.exp(x))**2
    integrated_square_error, _ = quad(squared_error, -1, 1)
    plt.title(r"Evaluation of Horners Method on $e^t$" + f"\n Error at x = 0: {error:.5f}, integrate square error: {integrated_square_error:.5f}")
    plt.plot(domain, approx, label="Approximate Values")
    plt.plot(domain, true, label = "True Values")
    plt.legend()
    plt.savefig("Horners")
    plt.show() 

if __name__ == "__main__":
    main()