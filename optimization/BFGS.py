import numpy as np 
from math import log10, floor
#import pandas as pd


def special(size):
    A = 2 * np.eye(size)
    lower = -1 * np.diag(np.ones(size-1), -1 )
    new = lower + lower.T
    return A + new



def f(x):
    n = x.shape[0]
    A = special(n)
    return x.T @ A @ x + 7


def f_prime(x):
    n = x.shape[0]
    A = special(n)
    return 2*A @ x + 7

def round_to_1(X):
   return  np.array([round(x, 2-int(floor(log10(abs(x))))) for x in X])


def BFGS(func, deriv, size):
    k = 1
    ks, xs,fxs, norms = [],[],[],[]
    H_inv_curr = np.eye(size) #Init Inverse Hessian 
    x_curr = np.ones(size) # Initialize x
    x_next = x_curr - deriv(x_curr) 
    s = (x_next - x_curr).reshape(-1,1) # Step Size 
    s_norm = np.linalg.norm(x_next - x_curr)
    ks.append(k), xs.append(x_next), norms.append(s_norm), fxs.append(func(x_next))
    while s_norm > 10e-10 and k < 1000:
        print(f"Iteration {k}\nx = {x_curr}\nAbsolute precision {s_norm}\n ")
        g = (deriv(x_next) - deriv(x_curr)).reshape(-1,1)
        A_prime = ((s.T@g + (g.T @H_inv_curr)@g)*(s@s.T))/(s.T@g)**2 
        B_prime = ((H_inv_curr@g)@s.T + s@(g.T@H_inv_curr))/(s.T @ g)
        H_inv_next =H_inv_curr + A_prime - B_prime 
        H_inv_curr = H_inv_next
        x_curr = x_next
        x_next = x_curr - H_inv_curr @ deriv(x_curr) 
        s = (x_next - x_curr).reshape(-1,1)
        s_norm = np.linalg.norm(s)
        k += 1
        ks.append(k), xs.append(round_to_1(x_next)), norms.append(s_norm),fxs.append(func(x_next))
    # dict1 = {"Iteration":ks,
    #  "Function Values":fxs,
    #   "Absolute Precision":norms}
    dict2 = {"Iteration": ks,
    "X":xs}
    # df1 = pd.DataFrame(index = dict1["Iteration"],data=dict1)
    df2 = pd.DataFrame(index = dict2["Iteration"],data=dict2)
    # df1.to_latex(buf = "table_1.tex", index=False)
    df2.to_latex(buf="table_2.tex", index=False)


    return x_curr



if __name__ == "__main__":
    root = BFGS(f, f_prime, 10)
    print(f"Root = {root}")
