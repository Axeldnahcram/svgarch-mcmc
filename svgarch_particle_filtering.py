import numpy as np
import scipy.stats
from scipy.stats import uniform
import scipy.optimize
from scipy.optimize import minimize
import math
import sympy
from sympy import DiracDelta
import matplotlib.pyplot as plt
import random


class SV_Garch_filtering(object):

    def __init__(self, y, gamma, alpha, beta, phi, M, T, malik=False):
        self.y = y
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.phi = phi
        self.T = T
        self.M = M
        self.V = np.zeros((T, M))
        self.W = np.zeros((T, M))
        self.Lambda = np.zeros((T, M))
        self.t = 0
        self.malik = malik
        self.initial_values()

    def initial_values(self):
        for i in range(self.M):
            self.V[0, i] = np.random.normal(0, 1, size=1) ** 2
        return self.V[0, :]

    def one_cycle(self):
        self.next_v()
        self.update_lambda()
        list_new_V = []
        for i in range(self.M):
            if self.malik:
                new_v = self.roll_malik_pitt()
            else:
                new_v = self.roll_discontinuous()
            list_new_V.append(new_v)
        self.V[self.t + 1, :] = np.array(list_new_V)
        self.t += 1

    def full_cycle(self):
        for i in range(self.T - 1):
            self.one_cycle()
        return self.V

    def next_v(self):
        t = self.t
        assert self.t < self.T
        for i in range(self.M):
            eps = self.y[t] / np.sqrt(self.V[t, i])
            xi = np.random.normal(0, 1, size=1)
            zeta = self.phi * eps + np.sqrt(1 - self.phi ** 2) * xi
            self.V[t + 1, i] = self.gamma + self.alpha * self.V[t, i] + self.beta * self.V[t, i] * (zeta) ** 2

    def update_lambda(self):
        t = self.t
        for i in range(self.M):
            self.W[t + 1, i] = ((2 * np.pi * self.V[t + 1, i]) ** (-0.5)) * np.exp(
                -0.5 * (self.y[t + 1] ** 2) / np.sqrt(self.V[t + 1, i]))
        for i in range(self.M):
            self.Lambda[t + 1, i] = self.W[t + 1, i] / sum(self.W[t + 1, :])

    def roll_discontinuous(self):
        t = self.t
        number = random.uniform(0, np.sum(self.Lambda[t + 1, 1:]))
        current = 0
        for i, bias in enumerate(self.Lambda[t + 1, :]):
            current += bias
            if number <= current:
                return self.V[t + 1, i]

    def roll_malik_pitt(self):
        t = self.t
        v_order = (-self.V[t + 1]).argsort()
        v_ordered = self.V[t + 1][v_order[::-1]]
        lbd_ordered = self.Lambda[t + 1][v_order[::-1]]
        trunkated_weights = [(lbd_ordered[i + 1] + lbd_ordered[i]) / 2 for i in range(0, self.M - 1)]

        weight = np.append(trunkated_weights, [lbd_ordered[-1] / 2])
        # print(v_ordered)
        x = np.random.uniform(v_ordered[0], v_ordered[-1])
        v_ordered = np.append(v_ordered, np.inf)
        tirage = self.tirage_sum(x, v_ordered, lbd_ordered)
        return tirage

    def tirage_uniform(self, x, v_ordered, i):
        v_ordered = np.append(v_ordered, np.inf)
        if x <= v_ordered[i + 1] and x >= v_ordered[i]:
            return uniform.cdf((x - v_ordered[i]) / (v_ordered[i + 1] - v_ordered[i]))
        else:
            return 0

    def tirage_sum(self, x, v_ordered, lambda_ordered):
        return sum([lambda_ordered[i] * self.tirage_uniform(x, v_ordered, i) for i in range(self.M)])

if __name__ == "__main__":
    y = np.array([0.3,0.2,0.5])
    theta = np.array([0.03, 0.5, 0.01, 0.05])
    gamma, alpha, beta, phi = theta[0], theta[1], theta[2], theta[3]
    M = 20
    T = 2
    l = SV_Garch_filtering(y, gamma, alpha, beta, phi, M, T, malik=False)
    m = l.full_cycle()
    print(f"the v is {l.V}")

