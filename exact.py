# Exact per-sample failure probability (publisher and client in different rows)
import math
from scipy.stats import binom
from newsetting import kappa, sigma, delta, W, B_P, closed


def pchunk(n, K1, K2, Q):
    h = math.ceil(kappa / Q)
    x = 1 / (K1 * K2)
    s = 2 * (1 - x) ** n - (1 - 2 * x) ** n
    a = 1 - 1 / K1
    b = 1 - 2 / K1
    for j in range(h):
        s += binom.pmf(j, n, 1 / K2) * (1 - 2 * a**j + b**j)
    return s


def pbound(n, K1, K2, Q):
    h = math.ceil(kappa / Q)
    return binom.cdf(h - 1, n, 1 / K2) + 2 * (1 - 1 / (K1 * K2)) ** n


def nmin(K1, K2, Q, d=delta, f=pchunk):
    lo, hi = 1, 10**6
    while lo < hi:
        m = (lo + hi) // 2
        if f(m, K1, K2, Q) <= d:
            hi = m
        else:
            lo = m + 1
    return lo


def qdelta(n, K1, K2, d=delta):
    for Q in range(1, 17):
        if pchunk(n, K1, K2, Q) <= d:
            return Q
    return None
