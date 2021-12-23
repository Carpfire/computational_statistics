import numpy as np 
#import pandas as pd


def legendre(n, x):
    if n == 1:
        return x
    if n == 0:
        if type(x) != np.ndarray :
            return 1
        else:
            return np.ones(x.shape) 
    else: return (2*n + 1)/(n + 1) *x * legendre(n-1, x) - n/(n + 1) * legendre(n - 2, x) 


def legendre_prime(n,x):
    if n == 1:
        return 1
    if n == 0:
            return 0
    else:
        return (n+1)*legendre(n-1, x) + x*legendre_prime(n-1, x)

def bisection(p, deg, interval):
    k = 0
    x_l, x_u = interval 
    sub = (x_u - x_l) / deg
    while x_u - x_l > sub:
        x_k = (x_u + x_l)/2
        k += 1
        if  np.sign(p(deg,x_k)) == np.sign(p(deg,x_l)): x_l = x_k
        else: x_u = x_k
    return x_l, x_u

def find_intervals(p, deg, interval):
    intervals = np.linspace(-1, 1, 48)
    solutions = []
    for i in range(1,48):
        x_l, x_u = bisection(p, deg, [intervals[i-1], intervals[i]])
        if np.sign(legendre(16, x_l)) != np.sign(legendre(16, x_u)):
            solutions.append([x_l, x_u])
    return solutions


def newtons(func, deriv, x0,n):
    k = 0
    x_next = x0 - func(n, x0)/deriv(n,x0)
    step = np.abs(x_next - x0)
    k += 1
    while step > 10e-10 and k < 10000: 
        x_prev = x_next
        x_next = x_prev - func(n,x_prev)/deriv(n,x_prev)
        step = np.abs(x_next - x_prev)
        k += 1 
        

    return x_next


def solve_roots(n):
    intervals = find_intervals(legendre, n, [-1,1])
    roots = [newtons(legendre, legendre_prime, interval[1], n) for interval in intervals]
    return roots

if __name__ == "__main__":
    
    all_roots = solve_roots(16)
    print(all_roots)
    # df = pd.DataFrame(data = all_roots)
    # df.to_latex(buf = "Roots_Table.tex")
