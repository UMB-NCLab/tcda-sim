# (O1) growth of the GET cost per sample (failed attempts included) with bad nodes, against the GET cost model.
import json, math, numpy as np
from eval_sim import attempt, network, kappa, pi
from scipy.stats import binom

K1, K2, Np, T = 15, 32, 2500, 4000
out = []


def P1P2(n, Q):
    h = math.ceil(kappa / Q)
    C = K1 * K2
    P1 = 2 * (1 - 1 / C) ** n - (1 - 2 / C) ** n
    s = sum(
        binom.pmf(j, n, 1 / K2) * (1 - 2 * (1 - 1 / K1) ** j + (1 - 2 / K1) ** j)
        for j in range(h)
    )
    return P1, s / (1 - P1)


for Q in [1, 4, 16, 32]:
    for r in [1, 2, 4, 8, 16]:
        rng = np.random.default_rng(Np + r * 100 + Q)
        L = []
        F = []
        La = []
        Fa = []
        for _ in range(T):
            G, B = network(rng, Np, K1, K2, r)
            R = 0
            tr = 0.0
            for i in range(200):
                c = rng.integers(K2)
                o = attempt(rng, G, B, c, Q)
                R += o["rounds"]
                tr += o["traffic"]
                if o["kind"] == "ok":
                    b = int(G[1, c])
                    La.append(o["rounds"])
                    Fa.append(o["traffic"] / b)
                    break
            L.append(R)
            F.append(tr / b)
        m = math.ceil(kappa / Q) - 1
        P1, P2 = P1P2(Np, Q)
        Nbar = r * Np / K2
        Lb = (
            2
            + math.log2(m * r + 1)
            + P1 / ((1 - P1) * (1 - P2))
            + P2 / (1 - P2) * (1 + math.ceil(math.log2(Nbar)))
        )
        out.append(
            dict(
                Q=Q,
                r=r,
                m=m,
                L=float(np.mean(L)),
                Lb=Lb,
                F=float(np.mean(F)),
                Fmin=(kappa - Q) * (1 / kappa + pi) + 1 + pi,
                Latt=float(np.mean(La)),
                Fatt=float(np.mean(Fa)),
                P1=P1,
                P2=P2,
            )
        )
        print({k: round(v, 3) for k, v in out[-1].items()}, flush=True)
json.dump(out, open("k64_growth.json", "w"), indent=1)
