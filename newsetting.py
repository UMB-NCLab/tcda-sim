# Numbers for the per-sample delta setting: sigma=32 MiB (M=256), K1=8, K2=32, delta=1/4
import math
from scipy.stats import binom
from scipy.optimize import brentq

kappa = 64
M = 128
D = 64
sym = 2048
sigma_B = M * M * sym
sigma = sigma_B / 2**20  # MiB
delta = 0.25
K1 = 8
K2 = 32
W = 131072
B_P = 25e6 * 4 / 8 / 2**20


def pchunk(n, K1, K2, Q):
    h = math.ceil(kappa / Q)
    return binom.cdf(h - 1, n, 1 / K2) + 2 * (1 - 1 / (K1 * K2)) ** n


def nmin(K1, K2, Q, d=delta):
    lo, hi = 1, 10**6
    while lo < hi:
        m = (lo + hi) // 2
        if pchunk(m, K1, K2, Q) <= d:
            hi = m
        else:
            lo = m + 1
    return lo


def closed(K1, K2, Q, d=delta):
    h = math.ceil(kappa / Q)
    L = math.log(2 / d)
    x = (
        L
        if h == 1
        else brentq(
            lambda x: x - (h - 1) * math.log(math.e * x / (h - 1)) - L,
            h - 1 + 1e-9,
            1e5,
        )
    )
    return math.ceil(max(K1 * K2 * math.log(4 / d), K2 * x))


def put(n, K2, Q, r=2):
    return r * n / K2 * (Q / kappa) * (sigma / K2)


if __name__ == "__main__":
    print("sigma MiB", sigma, "chunk MiB", sigma / K2, "B_P", round(B_P, 2))
    print("storage per node RDA GiB", sigma / K2 * W / 1024)
    for Q in range(1, 17):
        n = nmin(K1, K2, Q)
        print(
            "Q",
            Q,
            "h",
            math.ceil(kappa / Q),
            "N+",
            n,
            "closed",
            closed(K1, K2, Q),
            "PUT@min",
            round(put(n, K2, Q), 2),
        )
