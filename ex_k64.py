import math
from scipy.optimize import brentq
from scipy.stats import binom
from exact import pchunk, nmin

kappa = 64
K1, K2 = 15, 32
d = 0.25
r = 2


def closed(Q):
    h = math.ceil(kappa / Q)
    if h == 1:
        return math.ceil(K1 * K2 * math.log(2 / d))
    L = math.log(2 / d)
    x = brentq(
        lambda x: x - (h - 1) * math.log(math.e * x / (h - 1)) - L, h - 1 + 1e-9, 1e6
    )
    return math.ceil(max(K1 * K2 * math.log(4 / d), K2 * x))


print("Table example")
worst = 0
for Q in [64, 16, 8, 4, 2, 1]:
    n = nmin(K1, K2, Q)
    c = closed(Q)
    worst = max(worst, c / n - 1)
    print(Q, math.ceil(kappa / Q), n, c)
print("max over", round(worst, 3))
print(
    "Qdelta at N:",
    {
        n: min(q for q in range(1, 65) if pchunk(n, K1, K2, q) <= d)
        for n in [965, 966, 1000, 1222, 1500, 2000, 2221]
    },
)


def colfail(N):
    return sum(
        (-1) ** (j + 1) * math.comb(K2, j) * (1 - j / K2) ** N for j in range(1, K2 + 1)
    )


print("Table RDA")
for eps in [0.25, 1e-3, 1e-6, 1e-9]:
    N = 1
    while colfail(N) > eps:
        N += 1
    dd = pchunk(N, K1, K2, kappa)
    Q = min(q for q in range(1, 65) if pchunk(N, K1, K2, q) - dd <= 1e-4)
    print(eps, N, round(dd, 3), Q)
# Ethereum example at bootstrap
n = nmin(K1, K2, kappa)
N = 2 * n
mean = N / K2
q99 = binom.ppf(0.99, N, 1 / K2)
Bp = 25e6 * 4 / 8 / 2**20
TPUB = 3 * 32 * 2**20 * 8 / 1e9 + 0.06
Bnet = Bp * (1 - TPUB / 4)
print(
    "boot",
    n,
    "col mean",
    mean,
    "q99",
    q99,
    "RDA load/sigma",
    q99 / K2,
    "x budget",
    q99 / K2 * 32 / Bnet,
    "Bnet",
    Bnet,
    "Bfull",
    Bp,
)
dc = 1 - math.sqrt(1 - d)
LN = math.log(1 / dc)
for B in [Bnet, Bp]:
    smax = kappa * K2**2 * B / (r * (K1 * K2 - 1) * LN)
    print(
        "B",
        round(B, 2),
        "sigma_max",
        smax,
        "rda",
        smax / kappa,
        "N+ upper at 32MiB",
        kappa * K2**2 * B / (r * 32),
        "rda upper",
        K2**2 * B / (r * 32),
    )
print("lower bound", (K1 * K2 - 1) * LN)
