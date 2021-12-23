import numpy as np

#Our funnction with two imaginary roots 
def p(z):
    return np.power(z, 2) - 2*z + 2
    
#The derivative of the function 
def p_prime(z):
     return 2*z - 2 

#Newtons Method Implementation 
def newtons(func, deriv, x0):
    k = 0
    x_next = x0 - func(x0)/deriv(x0)
    step = np.abs(x_next - x0)
    k += 1
    while step > 1e-3 and k < 10000: 
        x_prev = x_next
        x_next = x_prev - func(x_prev)/deriv(x_prev)
        step = np.abs(x_next - x_prev)
        k += 1 

    return x_next, k


if __name__ == "__main__":

    z0_1 = -2 + -2j 
    root1, _ = newtons(p, p_prime, z0_1)

    z0_2 = 2 + 2j
    root2, _ = newtons(p, p_prime, z0_2)

    print(f'Root 1 is {root1}\nRoot 2 is {root2}')

