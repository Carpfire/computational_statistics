import numpy as np
import pandas as pd

def integrate_trapezoid(y, delta):
#Input: function evaluations, difference between grid points
#Output: Approximation of integral 
    first = y[:-1]
    second = y[1:]
    integrand = (first + second)
    integrand *= delta/2
    approx = integrand.sum()
    return approx 

def bessel_integrand(x, t):
    inner = x*np.cos(t)
    final = np.cos(inner)
    return final


def bessel_integrator(x):
    #Input: Bessel Function argument 
    #Output evaluation of bessel function using composite trapezoidal rule
    i = 3
    domain = np.linspace(0, 2*np.pi, i)
    delta = domain[1] - domain[0]

    y = bessel_integrand(x, domain)
    I1 = integrate_trapezoid(y, delta)/(2*np.pi)
    i += 1
    domain = np.linspace(0, 2*np.pi, i)

    y = bessel_integrand(x, domain)
    delta = domain[1] - domain[0]
    I2 = integrate_trapezoid(y, delta)/(2*np.pi)
    prec = abs(I2 - I1)
    I1 = I2
    while prec > 1e-12 and i < 1000:
        i+=1
        domain = np.linspace(0, 2*np.pi, i)
        y = bessel_integrand(x,domain)
        delta = domain[1] - domain[0]
        I2 = integrate_trapezoid(y, delta)/(2*np.pi)
        prec = abs(I2 - I1)
        I1=I2

    return I2, i
        

def main():
    out = np.ndarray((9,2))
    for i in range(9):
        x = 2**i
        print(x)
        val, points = bessel_integrator(x)
        out[i,:] = val, points

    print(out)
    table = pd.DataFrame(out).to_latex(index=False)
    table

    with open("problem3_table.tex", 'w') as f:
        f.write(table)
        f.close()


if __name__ == "__main__":
    main()

        




