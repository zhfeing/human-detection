import numpy as np
import matplotlib.pyplot as plt


class MixGaussianModule:
    def __init__(self):
        self._mu = np.array([0, 255], np.float)
        self._sigma = np.ones(2, np.float)*1000
        self._confidence = np.array([1/2, 1/2], np.float)

    def update_module(self, pixel):
        self._mu[0] = 1

    def get_module(self):
        return self._mu, self._sigma, self._confidence


if __name__ == "__main__":
    N = 100
    x1 = np.random.randn(N)*0.5 + 1
    x2 = np.random.randn(N)*2 + 4
    x = np.concatenate((x1, x2))

    plt.figure(0)
    plt.plot(x)

    mu = 0
    mx_2 = 10
    sigma_2 = mx_2 - mu**2
    alpha = 0.9

    mu_his = []
    sigma_2_his = []

    for i in range(2*N):
        mu = mu*alpha + x[i]*(1 - alpha)
        mx_2 = mx_2*alpha + (x[i]**2)*(1 - alpha)
        sigma_2 = mx_2 - mu**2

        mu_his.append(mu)
        sigma_2_his.append(sigma_2)

    mu_his = np.array(mu_his)
    sigma_2_his = np.array(sigma_2_his)

    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.plot(mu_his)
    plt.subplot(2, 1, 2)
    plt.plot(sigma_2_his)
    plt.show()
