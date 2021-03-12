print("hello world")
from usr_func import *
# import usr_func

# from skgstat import Variogram
# np.random.seed(20210309)
# np.random.seed(8004)
# np.random.seed(20210309)
# np.random.seed(20210309)


################################################################################################
# Section I : Set up grid
################################################################################################
# Discretise the grid
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
# plt.show()

################################################################################################
# Section I : Compute distance matrix
################################################################################################
# Compute the distance matrix
ddE = np.abs(sites1v * np.ones([1, n]) - np.ones([n, 1]) * sites1v.T)
dd2E = ddE * ddE
ddN = np.abs(sites2v * np.ones([1, n]) - np.ones([n, 1]) * sites2v.T)
dd2N = ddN * ddN
t = np.sqrt(dd2E + dd2N)
plotf(t, "distance matrix")

################################################################################################
# Section I : Compute covariance matrix
################################################################################################
# Compute the exponential covariance
eta = -3/20 # range coef in exponential kernel, 3/20 == .15
tau = 5 # iid noise for measuring, y = x + e(0, 5^2)

Sigma = Exp_cov(eta, t)  # matern covariance
plotf(Sigma, "matern cov")


################################################################################################
# Section I : Simulate the random field
################################################################################################
# Generate the realisation of the GRF
L = np.linalg.cholesky(Sigma)  # lower triangle matrix
x = np.dot(L, np.random.randn(n).reshape(-1, 1))
H = design_matrix(sites1v, sites2v) # different notation for the project
mu_prior = np.zeros([n, 1])
mu_real = mu_prior + x  # add covariance
plotf(np.copy(mu_real).reshape(n1, n2), "Realisation of the Norwegian Woods")


################################################################################################
# Section II : Sampling with variance reduction
################################################################################################
# design the sampling
M = n1
F = np.zeros([M, n])
# j = 20
mu_posterior = mu_real
Sigma_posterior = Sigma
T = np.identity(M) * tau ** 2  # T matrix for the measurement error

No_steps = 10
vr = np.zeros([No_steps, n2])

for k in range(No_steps):
    j_next, vr_temp = MVR_sampling(mu_posterior, Sigma_posterior, tau, M, n1, n2, n)
    vr[k, :] = vr_temp.T
    print("j next is ", j_next)
    F = np.zeros([M, n])
    for i in range(M):
        tempM = np.zeros([n1, n2])
        tempM[i, j_next] = True
        F[i, :] = np.ravel(tempM)

    y_sampled = np.dot(F, mu_posterior) + tau * np.random.randn(M).reshape(-1, 1)
    mu_posterior, Sigma_posterior = GRF2D(Sigma_posterior, F, T, y_sampled, mu_posterior)

    print(["posterior variance{:03d}".format(k)])
    plotf(np.diag(Sigma_posterior).reshape(n1, n2), "posterior variance_{:03d}".format(k))
    # plotf(mu_posterior.reshape(n1, n2), "posterior mean")
    # plt.figure(figsize=(5, 5))
    # plt.plot(vr_temp)
    # plt.ylim(0, .5)
    # plt.title("variance reduction at {:02d} iteration".format(k))
    # # plt.show()
    # plt.savefig("fig/vr_{:03d}".format(k) + ".png")

plt.close("all")


################################################################################################
# Section III : Sampling with sequential design based on VOI
################################################################################################
# static design: i.e. no adaptive selection, only one by one ish

T = np.identity(M) * tau ** 2 # measurement error matrix
VOI = np.zeros([M, 1])
for j in range(M):
    F = np.zeros([M, n])
    for i in range(M):
        tempM = np.zeros([n1, n2])
        tempM[i, j] = True
        F[i, :] = np.ravel(tempM)

    mu_w = np.sum(mu_prior)
    Cmatrix = np.dot(F, np.dot(Sigma, F.T)) + T
    Rj = np.dot(Sigma, np.dot(F.T, np.linalg.solve(Cmatrix, np.dot(F, Sigma))))
    r_wj = np.sqrt(np.sum(Rj))

    PoV = mu_w * norm.cdf(mu_w / r_wj) + r_wj * norm.pdf(mu_w / r_wj)
    PV = max(0, np.sum(mu_prior))

    # print("PoV is ", PoV)
    VOI[j] = PoV - PV
    # print("PV is ", PV)
    # print("VOI is ", VOI)

plt.plot(VOI, 'k*')
plt.title("VOI of 25 designs")
plt.show()
plt.grid()
plt.savefig("fig/voi.png")




