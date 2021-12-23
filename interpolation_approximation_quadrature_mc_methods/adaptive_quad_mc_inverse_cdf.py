from scipy.special import roots_legendre
from scipy.stats import beta 
import numpy as np
import scipy as sp
from tqdm import tqdm 
from itertools import chain 
import matplotlib.pyplot as plt 

def beta_cdf(x, a, b):
    rv = lambda t: beta.pdf(t, a, b)
    cdf = adaptive_quad(rv, (0, x), 1e-8)
    return cdf
    
def prob_in(x_1,x_2, a, b):
    rv = lambda t: beta.pdf(t, a, b)
    prob = adaptive_quad(rv, (x_1, x_2), 1e-8)
    return prob


def gl_integrator(func, bounds):
    lower, upper = bounds
    shift, scale = 0, 1
    if lower != -1 or upper != 1:
        A = np.array([[lower, 1], [upper, 1]])
        b = np.array([-1, 1])
        scale, shift = np.linalg.lstsq(A, b, rcond=-1)[0]
    roots, weights = roots_legendre(8)
    roots = (roots - shift)/scale
    evals = func(roots)
    approx = (evals*weights).sum()
    return approx/scale

def adaptive_quad(func, bounds, tol):
    I_init = gl_integrator(func, bounds)
    sub_1 = (bounds[0], (bounds[0]+bounds[1])/2)
    sub_2 = ((bounds[0]+bounds[1])/2, bounds[1])
    I_sub1 = gl_integrator(func, sub_1)
    I_sub2 = gl_integrator(func, sub_2)
    I_next = I_sub1 + I_sub2
    error = abs(I_init - I_next)
    if error > tol:
        return adaptive_quad(func, sub_1, tol) + adaptive_quad(func, sub_2, tol)
    else:
        return I_init


def inverse_cdf(n, a, b):
    uniform = sp.stats.uniform.rvs(size=n)
    f_prime = lambda x: beta.pdf(x, a, b)
    xs = []
    for u in uniform:
        x_prev = .5
        x_next = np.inf 
        error = abs(x_next - x_prev)
        add_term = 0
        itr = 1
        phi = beta_cdf(x_prev, a, b)
        while error > 1e-10 and itr < 1000:
            phi = phi + add_term
            phi_prime = f_prime(x_prev)
            x_next = x_prev - (phi - u)/phi_prime
            error = abs(x_next - x_prev)
            add_term = prob_in(x_prev, x_next, a, b)
            x_prev = x_next
            itr += 1 
        xs.append(x_prev)
    return np.array(xs)


def main():
    print(f"The value of F(.25, 3, 3) is {beta_cdf(.25, 3, 3)}")
    print(f"The value of F(.5, 4, 5) is {beta_cdf(.5, 4, 5)}")

#Split up ranges into different intervals for faster program
    means = []
    pbar = tqdm(range(1, 100))
    for n in pbar:
        true_mean = 5/(5+6)
        x = inverse_cdf(n,5, 6)
        means.append(true_mean - x.mean())

    pbar1 = tqdm(range(100, 1000, 10))
    for n in pbar1:
        true_mean = 5/(5+6)
        x = inverse_cdf(n,5, 6)
        means.append(true_mean - x.mean())
    pbar2 = tqdm(range(1000, 10000, 100))
    for n in pbar2:
        true_mean = 5/(5+6)
        x = inverse_cdf(n,5, 6)
        means.append(true_mean - x.mean())

    itr = list(chain(*(range(1, 100), range(100, 1000, 10), range(1000, 10000, 100))))
    
    plt.title("Deviation From True Mean v. Samples Size")
    plt.scatter(itr, means, s=8)
    plt.ylabel("Estimated Mean - True Mean")
    plt.xlabel("Sample Size")
    plt.show()

    best_fit = [-1 + -.5*np.log10(i) for i in itr]
    plt.title("Convergence rate of Monte Carlo Estimator ")
    plt.scatter(np.log10(itr), np.log10(means), label="Monte Carlo Estimates")
    plt.plot(np.log10(itr), best_fit, label="-1/2 slope line", c='r', linestyle= '--')
    plt.xlabel("Log N")
    plt.ylabel("Log Estimate")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()