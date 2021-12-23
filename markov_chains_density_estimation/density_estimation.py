import numpy as np 
from scipy.integrate import trapezoid
from scipy.stats import gamma
import matplotlib.pyplot as plt 
import random 

random.seed(10)

def kern(x, x_i, h):
    return np.where(((x-x_i)/h >=-1) & ((x-x_i)/h <= 1), (315/256)*(1-((x-x_i)/h)**2)**4, 0)


def optimize_h(x, true_pdf, samples, low, high):
    # f_hat_high = 1/50*sum(1/high*kern(x, samp, high) for samp in samples)
    # f_hat_low = 1/50*sum(1/low*kern(x, samp, low) for samp in samples)
    hs = []
    midpoint = .5*(high-low)
    prev_err = np.inf
    # f_hat_mid =  1/50*sum(1/midpoint*kern(x, samp, midpoint) for samp in samples)
    mid1, mid2 = (midpoint - low)/2, (high - midpoint)/2
    f_hat_mid1 = 1/50*sum(1/mid1*kern(x, samp, mid1) for samp in samples)
    err1 = np.sqrt(trapezoid(np.abs(true_pdf-f_hat_mid1)**2, x))
    f_hat_mid2 = 1/50*sum(1/mid2*kern(x, samp, mid2) for samp in samples)
    err2 = np.sqrt(trapezoid(np.abs(true_pdf-f_hat_mid2)**2, x))

    if err1 < err2:
        high = midpoint
        h = mid1
        hs.append(h)
    else:
        low = midpoint
        h = mid2
        hs.append(h)
    err_next = min(err1, err2)
    meta_err = abs(prev_err - err_next)
    itr = 1
    while meta_err > 1e-12 and itr < 1000:
        prev_err = err_next
        midpoint = .5*(high-low)
        mid1, mid2 = (midpoint - low)/2, (high - midpoint)/2
        f_hat_mid1 = 1/50*sum(1/mid1*kern(x, samp, mid1) for samp in samples)
        err1 = np.sqrt(trapezoid(np.abs(true_pdf-f_hat_mid1)**2, x))
        f_hat_mid2 = 1/50*sum(1/mid2*kern(x, samp, mid2) for samp in samples)
        err2 = np.sqrt(trapezoid(np.abs(true_pdf-f_hat_mid2)**2, x))
        if err1 < err2:
            high = midpoint
            h=mid1
            hs.append(h)
        else:
            low = midpoint
            h = mid2
            hs.append(h)
        
        err_next = min(err1, err2)
        meta_err = abs(prev_err - err_next)
        itr += 1

    return h, err_next, hs, itr


def main():
    x = np.linspace(0, 20, 50)
    samples = gamma(5).rvs(size=50)
    plt.title("histogram of samples")
    plt.hist(samples)
    plt.savefig("histogram")
    plt.close()
    true_pdf = gamma(5).pdf(x) 
    H = [.1, .25, .5, 1., 2.5,  5., 10.]
    estimators = [1/50*sum((1/h)*kern(x, samp, h) for samp in samples) for h in H]
    error = [np.sqrt(trapezoid(np.abs(true_pdf-f_hat)**2, x)) for f_hat in estimators]
    plt.title("Kernel Density Estimator of different bandwidths")
    for f_hat, h in zip(estimators, H):
        plt.plot(x, f_hat, label=f"Bandwidth={h}")
    plt.savefig("estimators")
    plt.close()
    plt.title("L2 error v. Bandwidth")
    plt.xlabel("Bandwidth")
    plt.ylabel("Error")
    plt.plot(H, error)
    plt.scatter(H[np.argmin(error)], min(error),c='r', label=f"Min Error {min(error)}")
    plt.legend()
    plt.savefig("errors")
    optim_h, err, hs, itr = optimize_h(x, true_pdf, samples, 0, 10)
    print(f"Optimized bandwidth is {optim_h}")


if __name__ == "__main__":
    main()