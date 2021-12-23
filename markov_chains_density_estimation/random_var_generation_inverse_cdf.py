import numpy as np 
from scipy.stats import uniform
from scipy.integrate import quad
from itertools import chain 
import matplotlib.pyplot as plt 


def exp_1_pdf(x):
    return np.exp(-x)


def exp_1_cdf(x):
    return 1 - np.exp(-x)

def chi2_2_cdf(x):
    return 1 - np.exp(-x/2)

def chi2_2_pdf(x):
    return .5*np.exp(-.5*x)

def inverse_cdf(pdf, n, cdf = None):
    us = uniform.rvs(size=n)
    xs = []
    for u in us:
        x_prev = .5
        x_next = np.inf 
        error = abs(x_next - x_prev)
        add_term = 0
        itr = 1
        if cdf != None:
            phi = cdf(x_prev)
        while error > 1e-10 and itr < 1000:
            phi = phi + add_term
            phi_prime = pdf(x_prev)
            x_next = x_prev - (phi - u)/phi_prime
            error = abs(x_next - x_prev)
            if cdf != None:
                add_term = cdf(x_next)-phi
            else:
                add_term = quad(pdf, x_prev, x_next)[0]
            x_prev = x_next
            itr += 1 
        xs.append(x_prev)
    return np.array(xs)


def generate_normal(n):
    exp_samps = inverse_cdf(exp_1_pdf,n, cdf=exp_1_cdf)
    chi2_samps = inverse_cdf(chi2_2_pdf,n, cdf=chi2_2_cdf)
    us1, us2 = uniform.rvs(size=n), uniform.rvs(size=n)
    R = np.sqrt(-2 *np.log(us1))
    theta = 2*np.pi*us2
    zs = R*np.cos(theta)
    zs = zs*chi2_samps + exp_samps
    return zs


def main():
    means = []
    true_mean = 1
    pbar = range(1, 100)
    for n in pbar:
        x = generate_normal(n)
        means.append(true_mean - x.mean())

    pbar1 = range(100, 1000, 10)
    for n in pbar1:
        x = generate_normal(n)
        means.append(true_mean - x.mean())
    pbar2 = range(1000, 10000, 100)
    for n in pbar2:
        x = generate_normal(n)
        means.append(true_mean - x.mean())
        
    itr = list(chain(*(range(1, 100), range(100, 1000, 10), range(1000, 10000, 100))))
    plt.title("Deviation From True Mean v. Samples Size")
    plt.scatter(itr, means, s=8)
    plt.ylabel("Estimated M\ean - True Mean")
    plt.xlabel("Sample Size")
    plt.savefig("Deviateion_MC")
    plt.close()
    best_fit = [-.5*np.log10(i) for i in itr]
    plt.title("Convergence rate of Monte Carlo Estimator ")
    plt.scatter(np.log10(itr), np.log10(means), label="Monte Carlo Estimates")
    plt.plot(np.log10(itr), best_fit, label="-1/2 slope line", c='r', linestyle= '--')
    plt.xlabel("Log N")
    plt.ylabel("Log Estimate")
    plt.legend()
    plt.savefig("Convergence_MC")



if __name__ == "__main__":
    main()

