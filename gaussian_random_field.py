print("hello world")
import numpy as np
import matplotlib.pyplot as plt
# from skgstat import Variogram
np.random.seed(8004)

## Functions used
def Matern_cov(sigma, eta, t):
    '''
    :param sigma: scaling coef
    :param eta: range coef
    :param t: distance matrix
    :return: matern covariance
    '''
    return sigma ** 2 * (1 + eta * t) * np.exp(-eta * t)

# def
def plotf(Y, string):
    plt.figure(figsize=(5,5))
    plt.imshow(Y)
    plt.title(string)
    plt.colorbar(fraction=0.045, pad=0.04)
    plt.gca().invert_yaxis()
    plt.show()

def design_matrix(sites1v, sites2v):
    '''
    :param sites1v:
    :param sites2v:
    :return:
    '''
    return np.hstack((np.ones([len(sites1v), 1]), sites1v, sites2v))

def mu(H, alpha):
    '''
    :param sites1v: grid along east direction
    :param sites2v: grid along north direction
    :param beta: regression coef
    :return: prior mean
    '''
    beta = np.hstack((-alpha, alpha, alpha))
    return np.dot(H, beta)

def sampling_design(n, M):
    '''
    :param n:
    :param M:
    :return:
    '''
    F = np.zeros([M, n])
    ind = np.random.randint(n, size = M)
    for i in range(M):
        F[i, ind[i]] = True
    return F, ind


# Setup the grid
n1 = 25 # number of grid points along east direction
n2 = 25 # number of grid points along north direction
n = n1 * n2 # total number of grid points

sites1 = np.arange(n1).reshape(-1, 1)
sites2 = np.arange(n2).reshape(-1, 1)
ww1 = np.ones([n1, 1])
ww2 = np.ones([n2, 1])
sites1 = sites1 * ww1.T
sites2 = ww2 * sites2.T

sites1v = sites1.flatten().reshape(-1, 1)
sites2v = sites2.flatten().reshape(-1, 1)

# Compute the distance matrix
ddE = np.abs(sites1v * np.ones([1, n]) - np.ones([n, 1]) * sites1v.T)
dd2E = ddE * ddE
ddN = np.abs(sites2v * np.ones([1, n]) - np.ones([n, 1]) * sites2v.T)
dd2N = ddN * ddN
t = np.sqrt(dd2E + dd2N)

plotf(t, "distance matrix")

# Simulate the initial random field
alpha = 1.0 # beta as in regression model
sigma = 1.0 # scaling coef in matern kernel
eta = .8 # range coef in matern kernel
# eta = 10 # range coef in matern kernel
tau = .05 # iid noise

BETA_TRUE = np.array([[-alpha], [alpha], [alpha]])
THETA_TRUE = np.array([[sigma], [eta], [tau]])

Sigma = Matern_cov(sigma, eta, t)  # matern covariance
# C_theta = Sigma + tau ** 2 * np.identity(n)
plotf(Sigma, "matern cov")
# plotf(C_theta, "C theta covariance")

L = np.linalg.cholesky(Sigma)  # lower triangle matrix
# L = np.linalg.cholesky(C_theta)  # lower triangle matrix

x = np.dot(L, np.random.randn(n).reshape(-1, 1))
H = design_matrix(sites1v, sites2v)
mu_prior = mu(H, alpha).reshape(n, 1)
plotf(np.copy(mu_prior).reshape(n1, n2), "prior mean")
mu_real = mu_prior + x  # add covariance
plotf(np.copy(mu_real).reshape(n1, n2), "realisation of grf")

# sampling from realisations
M = 200
F, ind = sampling_design(n, M)
G = np.dot(F, H)
y_sampled = np.dot(F, mu_real) + tau * np.random.randn(M).reshape(-1, 1)
x_ind, y_ind = np.unravel_index(ind, (n1, n2))
x_ind = x_ind.reshape(-1, 1)
y_ind = y_ind.reshape(-1, 1)

plt.figure()
plt.scatter(x_ind, y_ind, y_sampled, facecolors='none', edgecolors='k')
plt.title("random sample, circle size indicates true mean value")
plt.show()


##############################################
# #%% Compute the variogram
#
# V_v = Variogram(coordinates = np.hstack((x_ind, y_ind)), values = y_sampled.squeeze())
# V_v.plot(hist = False)
#
# print(V_v)
##############################################


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


# Use fisher scoring to find MLE parameters
# beta = np.zeros([3, 1])
beta = np.array([[-.9], [1.1], [1.2]])
theta = np.array([[.95], [.75], [.06]])
MAX_ITER = 1000
No_iter = 0
epsilon = 10
Beta = np.zeros([MAX_ITER, 3])
Likelihood = np.zeros([MAX_ITER, 1])

while No_iter < MAX_ITER and epsilon > .0001:

    C = C_matrix(theta)
    # Q = np.linalg.inv(C)
    beta = np.linalg.solve(np.dot(G.T, np.linalg.solve(C, G)), np.dot(G.T, np.linalg.solve(C, y_sampled)))
    Beta[No_iter, ] = beta.T
    z = y_sampled - np.dot(G, beta)
    Likelihood[No_iter, ] = -M/2 * np.log(2 * np.pi) -\
                            1/2 * np.log(np.linalg.det(C)) -\
                            1/2 * np.dot(z.T, np.linalg.solve(C, z))

    # Find dC*/dtheta
    dC_dSgm = dC_dsimga(theta)
    dC_dEta = dC_deta(theta)
    dC_dTau = dC_dtau(theta)

    u_sigma = -1/2 * np.trace(np.linalg.solve(C, dC_dSgm)) + \
              1/2 * np.dot(z.T, np.linalg.solve(C, np.dot(dC_dSgm, np.linalg.solve(C, z))))
    u_eta = -1 / 2 * np.trace(np.linalg.solve(C, dC_dEta)) + \
              1 / 2 * np.dot(z.T, np.linalg.solve(C, np.dot(dC_dEta, np.linalg.solve(C, z))))
    u_tau = -1 / 2 * np.trace(np.linalg.solve(C, dC_dTau)) + \
            1 / 2 * np.dot(z.T, np.linalg.solve(C, np.dot(dC_dTau, np.linalg.solve(C, z))))

    u = np.vstack((u_sigma, u_eta, u_tau))

    V11 = -1/2 * np.trace(np.linalg.solve(C, np.dot(dC_dSgm, np.linalg.solve(C, dC_dSgm))))
    V12 = -1 / 2 * np.trace(np.linalg.solve(C, np.dot(dC_dSgm, np.linalg.solve(C, dC_dEta))))
    V13 = -1 / 2 * np.trace(np.linalg.solve(C, np.dot(dC_dSgm, np.linalg.solve(C, dC_dTau))))
    V21 = -1 / 2 * np.trace(np.linalg.solve(C, np.dot(dC_dEta, np.linalg.solve(C, dC_dSgm))))
    V22 = -1 / 2 * np.trace(np.linalg.solve(C, np.dot(dC_dEta, np.linalg.solve(C, dC_dEta))))
    V23 = -1 / 2 * np.trace(np.linalg.solve(C, np.dot(dC_dEta, np.linalg.solve(C, dC_dTau))))
    V31 = -1 / 2 * np.trace(np.linalg.solve(C, np.dot(dC_dTau, np.linalg.solve(C, dC_dSgm))))
    V32 = -1 / 2 * np.trace(np.linalg.solve(C, np.dot(dC_dTau, np.linalg.solve(C, dC_dEta))))
    V33 = -1 / 2 * np.trace(np.linalg.solve(C, np.dot(dC_dTau, np.linalg.solve(C, dC_dTau))))

    V = np.array([V11, V12, V13, V21, V22, V23, V31, V32, V33]).reshape(3, 3)

    theta_new = theta - np.linalg.solve(V, u)  # here it is minus, but in the book, it says plus, needs to be rechecked
    epsilon = np.linalg.norm(theta_new - theta, 2) / np.linalg.norm(beta, 2)
    theta = theta_new
    print(epsilon)
    No_iter += 1

print(beta)
print(BETA_TRUE)
print(theta)
print(THETA_TRUE)

plt.plot(Likelihood[:No_iter], 'k')
plt.title('maximum likelihood function ')
plt.show()

alpha = sum(-beta[0], beta[1:2]) / 3
simgah, etah, tauh = theta

















