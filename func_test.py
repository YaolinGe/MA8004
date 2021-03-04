# def H(s, t): # define the t_{ij} (Euclidean) distance between two spatial locations
#     # H(s, t), s & t is the
#     ns = s.shape[0]
#     nt = t.shape[0]
#
#     Hs = [np.abs(i - j) for (i, j) in itertools.product(s[:, 0], t[:, 0])]
#     Hs = np.array(Hs).reshape(ns, nt)
#
#     Ht = [np.abs(i - j) for (i, j) in itertools.product(s[:, 1], t[:, 1])]
#     Ht = np.array(Ht).reshape(ns, nt)
#
#     return np.sqrt(Hs ** 2 + Ht ** 2)


