import numpy as np
from svgarch_particle_filtering import SV_Garch_filtering
from scipy.optimize import minimize
from matplotlib import pyplot as plt

def simulate_svgarch(gamma, alpha, beta, phi,  T): #alpha + beta < 1
    if alpha+beta>=1:
        raise NameError("Alpha + Beta must be less than 1")
    y = [0 for t in range(T)]
    v = [0 for t in range(T)]
    v[0] = gamma/(1 - alpha - beta)
    for t in range(T-1):
        eps = np.random.normal(0 , 1, size = 1)
        xi = np.random.normal(0,1,size=1)
        zeta = phi*eps+np.sqrt(1-phi**2)*xi
        y[t] = np.sqrt(v[t])*eps
        v[t+1] = gamma + alpha*v[t] + beta*v[t]*(zeta)**2
    return y, v


def neg_log_likelihood(theta, y, M, T): # theta = [gamma, alpha, beta, phi]
    gamma, alpha, beta, phi = theta[0], theta[1], theta[2], theta[3]
    l = SV_Garch_filtering(y, gamma, alpha, beta, phi, M, T, malik=False)
    m = l.full_cycle()
    t = np.sum(m, axis=0)
    return -np.sum(np.log(t))

def get_minimized_parameters(func, theta_0, method="Nelder-Mead", options=None):
    max_vrais = -minimize(func, theta_0,
                          method=method, options=options).x
    return max_vrais


if __name__ == "__main__":
    T = 200
    y, v = simulate_svgarch(gamma=0.01, alpha=0.925, beta=0.069, phi=0.1, T=200)
    plt.plot(v)
    plt.show()
