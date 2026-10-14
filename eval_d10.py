# Design evaluation with d_max=1/10 (RDA paper's example connectivity), Algorithm 1 with K1 scan.
import math, json, sys, os
from fractions import Fraction
from scipy.stats import binom
from exact import pchunk, nmin
from eval_sim import per_sample, kappa

M = 128
sigma = 32.0
W = 131072
Bs = float(os.environ.get("BS_MIB", 32 * 1024))
r = 2
DELTA = 0.25  # Bs: storage budget per node in MiB (32 GiB)
TPUB = 3 * 32 * 2**20 * 8 / 1e9 + 0.06
import os

DMAX = Fraction(os.environ.get("DMAX", "1/10"))
dc = 1 - math.sqrt(1 - DELTA)
LN = math.log(1 / dc)


def Nhat(n, K2, q99=True):
    return max(1.0, binom.ppf(0.99, r * n, 1 / K2)) if q99 else r * n / K2


def Bnet(Bp, q99=True):
    return Bp * (1 - TPUB / 4.0) if q99 else Bp


def K1min(K2):
    return math.ceil(1 / (DMAX - Fraction(1, K2)))


K2S = [d for d in range(1, M + 1) if M % d == 0 and Fraction(1, d) < DMAX]


def Qmax(n, K2, Bp, q99=True):
    qp = math.floor(kappa * Bnet(Bp, q99) * K2 / (Nhat(n, K2, q99) * sigma))
    qs = math.floor(kappa * Bs * K2 / (W * sigma))
    return min(kappa, qp, qs)


def P1P2(n, K1, K2, Q):
    h = math.ceil(kappa / Q)
    C = K1 * K2
    P1 = 2 * (1 - 1 / C) ** n - (1 - 2 / C) ** n
    s = sum(
        binom.pmf(j, n, 1 / K2) * (1 - 2 * (1 - 1 / K1) ** j + (1 - 2 / K1) ** j)
        for j in range(h)
    )
    return P1, s / (1 - P1)


def failterm(n, K1, K2, Q):
    P1, P2 = P1P2(n, K1, K2, Q)
    return P1 / ((1 - P1) * (1 - P2)) + P2 / (1 - P2) * (
        1 + math.ceil(math.log2(r * n / K2))
    )


def Lsucc_bound(Q):
    m = math.ceil(kappa / Q) - 1
    return 1 if Q == kappa else 2 + math.log2(2 * m + 1)


def bestK1(n, K2, Q):
    """scan K1 in [K1min,K1max]; feasibility is monotone in K1 (Thm 1), so stop at first infeasible"""
    best = None
    k1max = max(K1min(K2), math.floor((n / LN + 1) / K2))
    for K1 in range(K1min(K2), k1max + 1):
        if pchunk(n, K1, K2, Q) > DELTA:
            break
        f = failterm(n, K1, K2, Q)
        if best is None or f < best[0]:
            best = (f, K1)
    return None if best is None else best[1]


def candidates(n, Bp, Qfix=None, q99=True):
    out = []
    for K2 in K2S:
        Q = Qmax(n, K2, Bp, q99) if Qfix is None else Qfix
        if Qfix is not None and Qmax(n, K2, Bp, q99) < Qfix:
            continue
        if Q < 1:
            continue
        K1 = bestK1(n, K2, Q)
        if K1 is None:
            continue
        out.append((K1, K2, Q))
    return out


def feasible(n, Bp, rda=False):
    for K2 in K2S:
        Q = Qmax(n, K2, Bp)
        if Q < 1 or (rda and Q < kappa):
            continue
        if pchunk(n, K1min(K2), K2, Q) <= DELTA:
            return True
    return False


def ranges(Bp, rda=False, Nmax=40000):
    iv = []
    start = None
    for n in range(1, Nmax + 1):
        o = feasible(n, Bp, rda)
        if o and start is None:
            start = n
        if not o and start is not None:
            iv.append((start, n - 1))
            start = None
    if start is not None:
        iv.append((start, Nmax))
    return iv


if __name__ == "__main__":
    print("K2S", K2S, "K1min", [K1min(k) for k in K2S])
    res = {"grids": [(K1min(k), k) for k in K2S], "ranges": {}}
    for mb in [25, 50, 100]:
        Bp = mb * 1e6 * 4 / 8 / 2**20
        res["ranges"][mb] = dict(tcda=ranges(Bp), rda=ranges(Bp, rda=True))
        print(mb, res["ranges"][mb], flush=True)
    TAG = os.environ.get("DMAX", "1/10").replace("/", "_")
    json.dump(res, open("k64_ranges_d" + TAG + ".json", "w"), indent=1)
