import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


x = np.array([0, .6981317, 1.3962634, 2.0943951, 2.7925268, 3.4906585, 4.1887902, 4.88692191, 5.58505361, 6.28318531])
y = np.array([-0.00799366, .50389564, 0.92653312, .7628036, .37189376, -.1965461, -.93017225, -1.04932639, -.72417058,-.07469816])
x_star = np.linspace(0, 2*np.pi, 100)
sigma = .01

def k(x_star,x):
    x_prime = np.stack([x for i in range(len(x_star))], axis=0)
    x = x_star.reshape(-1, 1) - x_prime
    return np.exp(-(x)**2) 


def mean_posterior(k, sigma, x, x_star, y):
    cov1 = k(x_star, x)
    cov2 = np.linalg.inv((k(x, x) + sigma**2*np.eye(x.shape[0])))
    res = cov1 @ (cov2 @ y)
    return res
    

def cov_posterior(k, sigma, x, x_star, y):
    cov1 = k(x_star, x_star)
    cov2 = k(x, x)
    cov3 = k(x_star, x)
    cov4 = cov3.T
    cov2 = np.linalg.inv(cov2 + sigma**2 * np.eye(x.shape[0]))
    return cov1 - cov3 @ cov2 @ cov4


def mean_0(x):
    return 0


def gp_regression(kernel, x_new, x, y, sigma):
    cov_post = cov_posterior(kernel, sigma, x, x_new, y)
    mean_post = mean_posterior(kernel, sigma, x, x_new, y)
    pred_dist = multivariate_normal(mean = mean_post,cov= cov_post, allow_singular=True)  
    return pred_dist


if __name__ == "__main__":
    
    post_pred_zero = gp_regression(k, x_star, x, y, sigma)
    post_pred_sin = gp_regression(k, x_star, x, y - np.sin(x), sigma)
    post_pred_sin.mean = post_pred_sin.mean + np.sin(x_star)
    plt.figure(1)
    plt.plot(x_star, post_pred_zero.mean, '--')
    plt.title("GP Regression EV With 0 Mean Function")
    plt.plot(x_star, post_pred_sin.mean, '--')
    plt.title("GP Regression EV with Sin Mean Function")
    plt.figure(2)
    draws = post_pred_zero.rvs(20)
    plt.plot(x_star, draws.T, '--')
    plt.title("20 draws from posterior predictive distribution")
    plt.show()
