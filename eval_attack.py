# Attack experiments: (1) PeerDAS custody-sampling under Sybils vs TCDA/RDA; (2) designated vs random coefficients
# (crowding) ablation; (3) targeted placement of bad nodes (concentrated in one column).
import numpy as np, math, json
from exact import pchunk
from eval_sim import collector, kappa, pi

rng0 = np.random.default_rng(11)


# ---------- (1) PeerDAS custody sampling (Fulu): 128 columns, honest nodes sample >= 8 groups via gossip.
# Model: to sample column j, an honest node contacts C distinct peers that advertise custody of j (its gossip mesh,
# D=8, plus req/resp fallbacks), drawn uniformly from all advertisers. Honest advertisers of j: each honest node
# custodies/samples 8 of 128 columns. Sybils advertise custody of every column (cgc=128, self-reported).
# The sample fails iff every contacted advertiser is a Sybil (optimistic for PeerDAS: every honest custodian has it).
def peerdas_fail(Np, Nb, C=16, T=100000, rng=rng0, s=8, cols=128):
    H = rng.binomial(Np, s / cols, size=T)  # honest advertisers of the sampled column
    tot = H + Nb
    fail = np.zeros(T, bool)
    for i in range(T):
        if H[i] == 0:
            fail[i] = True
            continue
        k = min(C, tot[i])
        # hypergeometric: number of honest among k drawn without replacement
        fail[i] = rng.hypergeometric(H[i], Nb, k) == 0 if Nb > 0 else False
    return fail.mean()


def peerdas_fail_exact(Np, Nb, C=16, s=8, cols=128):
    from scipy.stats import binom

    tot = 0.0
    for h in range(0, Np + 1):
        w = binom.pmf(h, Np, s / cols)
        if w < 1e-15:
            continue
        if h == 0:
            tot += w
            continue
        k = min(C, h + Nb)
        # P(no honest among k draws without replacement)
        p = 1.0
        for t in range(k):
            p *= max(Nb - t, 0) / (h + Nb - t)
        tot += w * p
    return tot


# ---------- (2)+(3) grid simulator with bad counts per cell and optional attacks
def network_cells(rng, Np, K1, K2, Nb, target_col=None, frac_target=0.0):
    G = rng.multinomial(Np, [1 / (K1 * K2)] * (K1 * K2)).reshape(K1, K2)
    nt = int(round(frac_target * Nb)) if target_col is not None else 0
    B = rng.multinomial(Nb - nt, [1 / (K1 * K2)] * (K1 * K2)).reshape(K1, K2)
    if nt:
        B[:, target_col] += rng.multinomial(nt, [1 / K1] * K1)
    return G, B


def attempt_cells(rng, G, B, c, Q, coeff="designated"):
    a = G[0, c]
    b = G[1, c]
    n = int(G[:, c].sum())
    Nc = n + int(B[:, c].sum())
    if b == 0 or a == 0:
        return "cell", 1
    if Q == kappa:
        return "ok", 1
    if coeff == "random" and B[0, c] > 0:
        # crowding: a bad relay of the publisher's cell sends every column node the same Q valid pieces first;
        # good nodes then span a Q-dim space and no collector can reach rank kappa.
        ts = [collector(rng, n - 1, Nc - 1, Q)[0] for _ in range(b)]
        return "short", 1 + max(math.ceil(math.log2(Nc)), 1)
    ts = []
    ok = False
    for _ in range(b):
        t, q, coords, s = collector(rng, n - 1, Nc - 1, Q)
        ts.append(t)
        ok |= s
    return ("ok", 1 + min(ts)) if ok else ("short", 1 + max(ts))


def fail_rate(
    Np, K1, K2, Q, Nb, coeff="designated", T=20000, seed=1, target=False, frac=0.0
):
    rng = np.random.default_rng(seed)
    f = 0
    lat = []
    latc = []
    for _ in range(T):
        tc = 0 if target else None
        G, B = network_cells(rng, Np, K1, K2, Nb, tc, frac)
        c = (
            0 if target else rng.integers(K2)
        )  # targeted: evaluate samples landing in the attacked column
        k, r = attempt_cells(rng, G, B, c, Q, coeff)
        if k != "ok":
            f += 1
        else:
            lat.append(r)
    return f / T, (np.mean(lat) if lat else float("nan"))


if __name__ == "__main__":
    out = {}
    # (1) PeerDAS vs grid designs as Sybils grow, N+=1000 honest
    Np = 1000
    rows = []
    for ratio in [0, 0.1, 0.25, 0.5, 1, 2, 5, 10]:
        Nb = int(ratio * Np)
        pd8 = peerdas_fail_exact(Np, Nb, C=8)
        pd16 = peerdas_fail_exact(Np, Nb, C=16)
        rows.append(
            dict(
                ratio=ratio,
                peerdas8=pd8,
                peerdas16=pd16,
                tcda=pchunk(Np, 8, 32, 4),
                rda=pchunk(Np, 8, 32, 16),
            )
        )
        print(rows[-1], flush=True)
    # MC check of the PeerDAS formula at two points
    for ratio in [0.5, 2]:
        print(
            "mc check",
            ratio,
            peerdas_fail(Np, int(ratio * Np), C=16, T=40000),
            peerdas_fail_exact(Np, int(ratio * Np), C=16),
        )
    out["peerdas"] = rows
    # (2) crowding ablation on (8,32), Q=4, N+ from 514..1000, r=2
    abl = []
    for n in [514, 647, 800, 1000]:
        for coeff in ["designated", "random"]:
            p, _ = fail_rate(n, 8, 32, 4, n, coeff, T=20000, seed=n)
            abl.append(dict(N=n, coeff=coeff, p=p, exact=pchunk(n, 8, 32, 4)))
            print(abl[-1], flush=True)
    out["crowding"] = abl
    # (3) targeted placement: fraction f of N- bad nodes concentrated in column 0; samples in column 0
    tgt = []
    n = 1000
    Nb = 1000
    for frac in [0, 0.25, 0.5, 1.0]:
        for Q in [4, 16]:
            p, l = fail_rate(
                n,
                8,
                32,
                Q,
                Nb,
                "designated",
                T=20000,
                seed=int(frac * 100) + Q,
                target=True,
                frac=frac,
            )
            Ncol = n / 32 + Nb * (1 - frac) / 32 + Nb * frac
            put = Q / kappa * Ncol * 32 / 32  # MiB per relay, sigma=32 MiB, K2=32
            tgt.append(
                dict(
                    frac=frac,
                    Q=Q,
                    p=p,
                    exact=pchunk(n, 8, 32, Q),
                    Latt=l,
                    Ncol=Ncol,
                    put_MiB=put,
                )
            )
            print(tgt[-1], flush=True)
    out["targeted"] = tgt
    json.dump(out, open("eval_attack.json", "w"), indent=1, default=float)
