import numpy as np
import matplotlib.pyplot as plt
from random import sample
from scipy.stats import norm

## Functions used
def Matern_cov(sigma, eta, t):
    '''
    :param sigma: scaling coef
    :param eta: range coef
    :param t: distance matrix
    :return: matern covariance
    '''
    return sigma ** 2 * (1 + eta * t) * np.exp(-eta * t)


def Exp_cov(eta, t):
    '''
    :param eta:
    :param t:
    :return:
    '''
    return np.exp(eta * t)


# def
def plotf(Y, string):
    '''
    :param Y:
    :param string:
    :return:
    '''
    plt.figure(figsize=(5,5))
    plt.imshow(Y)
    plt.title(string)
    plt.colorbar(fraction=0.045, pad=0.04)
    plt.gca().invert_yaxis()
    # plt.show()
    plt.savefig("fig/"+string+".png")


def design_matrix(sites1v, sites2v):
    '''
    :param sites1v:
    :param sites2v:
    :return:
    '''
    return np.hstack((np.ones([len(sites1v), 1]), sites1v, sites2v))


def mu(H, beta):
    '''
    :param H: design matrix
    :param beta: regression coef
    :return: prior mean
    '''
    # beta = np.hstack((-alpha, alpha, alpha))
    return np.dot(H, beta)


def sampling_design(n, M):
    '''
    :param n:
    :param M:
    :return:
    '''
    F = np.zeros([M, n])
    # ind = np.random.randint(n, size = M)
    ind = sample(range(n), M)
    for i in range(M):
        F[i, ind[i]] = True
    return F, ind


def sampling_design_VOI(n, M, n1, n2, ind):
    return 0


def GRF2D(Sigma, F, T, y_sampled, mu_prior):
    '''
    :param Sigma:
    :param F:
    :param T:
    :param y_sampled:
    :param mu_prior:
    :return:
    '''
    Cmatrix = np.dot(F, np.dot(Sigma, F.T)) + T
    mu_posterior = mu_prior + np.dot(Sigma, np.dot(F.T, np.linalg.solve(Cmatrix, (y_sampled - np.dot(F, mu_prior)))))
    Sigma_posterior = Sigma - np.dot(Sigma, np.dot(F.T, np.linalg.solve(Cmatrix, np.dot(F, Sigma))))
    return (mu_posterior, Sigma_posterior)


def MVR_sampling(mu_prior, Sigma_prior, tau, M, n1, n2, n):
    '''
    :param mu_prior:
    :param Sigma_prior:
    :param T:
    :param M:
    :param n1:
    :param n2:
    :param n:
    :return:
    '''
    T = np.identity(M) * tau ** 2  # T matrix for the measurement error
    vr = np.zeros([M, 1])
    for j in range(n2):
        F = np.zeros([M, n])
        for i in range(M):
            tempM = np.zeros([n1, n2])
            tempM[i, j] = True
            F[i, :] = np.ravel(tempM)

        y_sampled = np.dot(F, mu_prior) + tau * np.random.randn(M).reshape(-1, 1)
        mu_posterior, Sigma_posterior = GRF2D(Sigma_prior, F, T, y_sampled, mu_prior)
        vr[j] = np.mean(np.diag(Sigma_posterior))
    mvr_ind = np.argmin(vr)
    return mvr_ind, vr


def PoV(mu_posterior, Sigma_posterior, T, F):
    '''
    :param mu_posterior:
    :param Sigma_posterior:
    :param T:
    :param F:
    :return:
    '''
    mu_w = np.sum(mu_posterior)
    Cmatrix = np.dot(F, np.dot(Sigma_posterior, F.T)) + T
    Rj = np.dot(Sigma_posterior, np.dot(F.T, np.linalg.solve(Cmatrix, np.dot(F, Sigma_posterior))))
    r_wj = np.sqrt(np.sum(Rj))
    PoV = mu_w * norm.cdf(mu_w / r_wj) + r_wj * norm.pdf(mu_w / r_wj)
    return PoV


def PV(mu_posterior):
    '''
    :param mu_posterior:
    :return:
    '''
    return max(0, np.sum(mu_posterior))


def VOI(pov, pv):
    '''
    :param PoV:
    :param PV:
    :return:
    '''
    return pov - pv


def VOI_sampling(Price, mu_prior, Sigma_prior, tau, M, n, n1, n2):
    '''
    :param mu_prior:
    :param Sigma_prior:
    :param tau:
    :param M:
    :param n:
    :param n1:
    :param n2:
    :return:
    '''
    T = np.identity(M) * tau ** 2  # T matrix for the measurement error
    voi = np.zeros([M, 1])
    for j in range(n2):
        F = np.zeros([M, n])
        for i in range(M):
            tempM = np.zeros([n1, n2])
            tempM[i, j] = True
            F[i, :] = np.ravel(tempM)

        pov = PoV(mu_prior, Sigma_prior, T, F)
        pv = PV(mu_prior)
        voi[j] = VOI(pov, pv)

    if np.max(voi) >= Price:
        voi_ind = np.argmax(voi)
    else:
        voi_ind = -1
    return voi_ind, voi


###################### other functions used for kriging and parameter estimation ##############################
###############################################################################################################


def C_matrix(theta):
    '''
    :param theta:
    :return:
    '''
    sigma, eta, tau = theta
    Sigma = Matern_cov(sigma, eta, t)
    C = np.dot(F, np.dot(Sigma, F.T)) + np.identity(F.shape[0]) * tau ** 2
    return C


def dC_dsimga(theta):
    '''
    :param theta:
    :return:
    '''
    sigma, eta, tau = theta
    Km = Matern_cov(1.0, eta, t) # t here is the distance matrix, H is the design matrix, similar to X
    dC_dsgm = np.dot(F, np.dot(Km, F.T))
    return dC_dsgm


def dC_deta(theta):
    '''
    :param theta:
    :return:
    '''
    sigma, eta, tau = theta
    Kn = sigma ** 2 * (-eta * t) * np.exp(-eta * t)
    return np.dot(F, np.dot(Kn, F.T))


def dC_dtau(theta):
    '''
    :param theta:
    :return:
    '''
    return np.identity(F.shape[0])
