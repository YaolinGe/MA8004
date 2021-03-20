print("hello world")
import numpy as np
import matplotlib.pyplot as plt
from random import sample

# from skgstat import Variogram
np.random.seed(20210309)
# np.random.seed(8004)
# np.random.seed(20210309)
# np.random.seed(20210309)

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
    fig = plt.gcf()
    plt.imshow(Y)
    plt.title(string)
    plt.xlabel("s1")
    plt.ylabel("s2")
    plt.colorbar(fraction=0.045, pad=0.04)
    plt.gca().invert_yaxis()
    plt.show()
    return fig

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

# def center_sampling_design(n, n1, n2, M):


# Setup the grid
n1 = 25 # number of grid points along east direction
n2 = 25 # number of grid points along north direction
n = n1 * n2 # total number of grid points

dn1 = 1/n1
dn2 = 1/n2
sites1 = np.arange(0, 1, dn1).reshape(-1, 1)
sites2 = np.arange(0, 1, dn2).reshape(-1, 1)
ww1 = np.ones([n1, 1])
ww2 = np.ones([n2, 1])
sites1m = sites1 * ww1.T # sites1m is the matrix version of sites1
sites2m = ww2 * sites2.T

sites1v = sites1m.flatten().reshape(-1, 1) # sites1v is the vectorised version
sites2v = sites2m.flatten().reshape(-1, 1)

# plt.figure(figsize=(5, 5))
# plt.plot(sites1v, sites2v, 'k.')
# plt.xlabel("s1")
# plt.ylabel("s2")
# plt.title("grid decomposition")
# plt.show()


# Compute the distance matrix
ddE = np.abs(sites1v * np.ones([1, n]) - np.ones([n, 1]) * sites1v.T)
dd2E = ddE * ddE
ddN = np.abs(sites2v * np.ones([1, n]) - np.ones([n, 1]) * sites2v.T)
dd2N = ddN * ddN
t = np.sqrt(dd2E + dd2N)

plotf(t, "distance matrix")

# Simulate the initial random field
alpha = 1.0 # beta as in regression model
sigma = 0.25  # scaling coef in matern kernel
eta = 9 # range coef in matern kernel
# eta = 10 # range coef in matern kernel
tau = .0025 # iid noise

beta1 = -2
beta2 = 3
beta3 = 1

BETA_TRUE = np.array([[beta1], [beta2], [beta3]])
THETA_TRUE = np.array([[sigma], [eta], [tau]])

Sigma = Matern_cov(sigma, eta, t)  # matern covariance
# C_theta = Sigma + tau ** 2 * np.identity(n)
plotf(Sigma, "matern cov")
# plotf(C_theta, "C theta covariance")

L = np.linalg.cholesky(Sigma)  # lower t    riangle matrix
# L = np.linalg.cholesky(C_theta)  # lower triangle matrix

x = np.dot(L, np.random.randn(n).reshape(-1, 1))
H = design_matrix(sites1v, sites2v) # different notation for the project
mu_prior = mu(H, BETA_TRUE).reshape(n, 1)
fig = plotf(np.copy(mu_prior).reshape(n1, n2), "True mean of the field trend")
fig.savefig("fig_presentation/True_mean.pdf")

mu_real = mu_prior + x  # add covariance
fig = plotf(np.copy(mu_real).reshape(n1, n2), "Realisation of the random field")
fig.savefig("fig_presentation/Realisation.pdf")


#%%
# sampling from realisations
M = 200
F, ind = sampling_design(n, M)
G = np.dot(F, H)
y_sampled = np.dot(F, mu_real) + tau * np.random.randn(M).reshape(-1, 1)
x_ind, y_ind = np.unravel_index(ind, (n1, n2))

x_ind = sites1[x_ind]
y_ind = sites2[y_ind]

plt.figure(figsize=(5, 5))
plt.scatter(x_ind, y_ind, 10 * (y_sampled - np.amin(y_sampled)), facecolors='none', edgecolors='k')
plt.xlabel("s1")
plt.ylabel("s2")
# y_sampled times 10 for scaling in the plot
# abs() added here to make the scatter plot work properly
plt.title("Random observation design")
plt.savefig("fig_presentation/Random_design.pdf")
plt.show()


# # sampling based on center design
# Mc = n1 * n2
# Fc = np.zeros([Mc, n])
# for i in range(n1):
#     temp = np.zeros([n1, n2])
#     temp[i, 12] = True
#     Fc[i, :] = np.ravel(temp)
#
# for j in range(n2):
#     # if j == 12:
#     #     continue
#     temp = np.zeros([n1, n2])
#     temp[12, j] = True
#     Fc[j+n1, :] = np.ravel(temp)
#
# Gc = np.dot(Fc, H)
# y_sampled_c = np.dot(Fc, mu_real) + tau * np.random.randn(Mc).reshape(-1, 1)
#
# plt.figure(figsize=(5, 5))
# plt.scatter(10 * (y_sampled_c - np.amin(y_sampled_c)), facecolors='none', edgecolors='k')
# plt.xlabel("s1")
# plt.ylabel("s2")
# # y_sampled times 10 for scaling in the plot
# # abs() added here to make the scatter plot work properly
# plt.title("Random observation design")
# plt.savefig("fig_presentation/Center_design.pdf")
# plt.show()

#%%
##############################################
#%% Compute the variogram
#
V_v = Variogram(coordinates = np.hstack((x_ind, y_ind)), values = y_sampled.squeeze())
V_v.plot(hist = False)
#
print(V_v)
##############################################


#%%
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
beta = np.array([[-2.1], [3.1], [.9]])
theta = np.array([[.245], [9.3], [.003]])
MAX_ITER = 5000
No_iter = 0
epsilon = 10
Beta = np.zeros([MAX_ITER, 3])
Likelihood = np.zeros([MAX_ITER, 1])

while No_iter < MAX_ITER and epsilon > .0001:

    C = C_matrix(theta)
    beta = np.linalg.solve(np.dot(G.T, np.linalg.solve(C, G)), np.dot(G.T, np.linalg.solve(C, y_sampled)))
    Beta[No_iter, ] = beta.T
    z = y_sampled - np.dot(G, beta)
    lik = -M/2 * np.log(2 * np.pi) -\
        [1/2 * np.log(np.linalg.det(C)) if np.linalg.det(C) != 0 else 0] - \
        1/2 * np.dot(z.T, np.linalg.solve(C, z)) # otherwise, it becomes inf
    Likelihood[No_iter, ] = lik if lik is not np.inf else 0

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
    print(epsilon , " , iter no is ", No_iter)
    No_iter += 1

# print(beta)
# print(BETA_TRUE)
# print(theta)
# print(THETA_TRUE)

plt.plot(Likelihood[:No_iter], 'k')
plt.title('maximum likelihood function ')
plt.show()

# alphah = (sum(np.abs(beta)) / 3).squeeze()
thetah = theta
sigmah = theta[0].squeeze()
etah = theta[1].squeeze()
tauh = theta[2].squeeze()
betah = beta
beta1 = beta[0].squeeze()
beta2 = beta[1].squeeze()
beta3 = beta[2].squeeze()

# print('Estimated sigma is ', sigmah, "\nEstimated eta is ", etah, \
#       "\nEstimated tau is ", tauh, "\nEstimated alpha is ", alphah)
print("\nEstimated sigma is ", np.round(sigmah, 3), "; True sigma is ", THETA_TRUE[0].squeeze(), \
      "\nEstimated eta is ", np.round(etah,2), "; True eta is ", THETA_TRUE[1].squeeze(), \
      "\nEstimated tau is ", np.round(tauh,5), "; True tau is ", THETA_TRUE[2].squeeze(), \
      "\nEstimated beta1 is ", np.round(beta1, 2), "; True beta1 is ", BETA_TRUE[0].squeeze(), \
      "\nEstimated beta2 is ", np.round(beta2, 2), "; True beta2 is ", BETA_TRUE[1].squeeze(), \
      "\nEstimated beta3 is ", np.round(beta3, 2), "; True beta3 is ", BETA_TRUE[2].squeeze())


#%% Kriging part
Sigmah = Matern_cov(sigmah, etah, t) # estimated covariance matrix
Lh = np.linalg.cholesky(Sigmah) #h here refers to hat
mh = mu(H, betah).reshape(-1, 1) + np.dot(Lh, np.random.randn(n).reshape(-1, 1))
Ch = C_matrix(thetah)
xp = mh + np.dot(Sigmah, np.dot(F.T, np.linalg.solve(C, (y_sampled - np.dot(F, mh)))))
plotf(xp.reshape(n1, n2), "posterior mean")
Sigmap = Sigmah - np.dot(Sigmah, np.dot(F.T, np.linalg.solve(Ch, np.dot(F, Sigmah))))
# plotf(Sigmap, "posterior covariance")
estd = np.sqrt(np.diag(Sigmap)).reshape(-1, 1)
plotf(estd.reshape(n1, n2), "posterior std")
MSE = np.sqrt(np.sum(abs(xp - mu_real) ** 2) / n)
print("The prediction error is ", MSE)











