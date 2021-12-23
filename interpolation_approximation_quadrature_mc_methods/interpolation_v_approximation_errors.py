from numpy.polynomial import legendre, chebyshev
from scipy.integrate import quad 
import numpy as np
import matplotlib.pyplot as plt 


def main():
    domain = np.linspace(0, np.pi)
    y = np.cos(domain)
    ns = [1, 2, 4, 8, 16]
    for n in ns:
        coef = legendre.legfit(domain, y, n)
        L = legendre.Legendre(coef)
        C = chebyshev.Chebyshev.interpolate(np.cos, n, [0, np.pi])
        L_y = L(domain)
        C_y = C(domain)
        integrandL = lambda t: np.abs(L(t) - np.cos(t))**2
        integrandC = lambda t: np.abs(C(t) - np.cos(t))**2
        L2_L = np.sqrt(quad(integrandL, 0, np.pi))
        L2_C = np.sqrt(quad(integrandC, 0, np.pi))
        plt.title(f"Legendre Approximation v. Chebyshev Interpolation, degree {n}")
        plt.plot(domain, L_y, label = f"Legendre Polynomial, L2 Error {L2_L[0]:.6f}")
        plt.plot(domain, C_y, label= f"Chebyshev Polynomial, L2 Error: {L2_C[0]:.6f}")
        plt.plot(domain, y)
        plt.legend()
        plt.savefig(f"polynomials{n}")
        plt.close()

        

if __name__ == "__main__":
    main()