# (O2) Algorithm 1 over the two design inputs: good nodes N+ and population bound N_max = rho*N+ (25 Mbit/s, d_max=1/10).
import json, math, sys
import eval_d10 as E
from eval_sim import per_sample, kappa

Bp = 25 * 1e6 * 4 / 8 / 2**20
NS = [
    600,
    700,
    868,
    1000,
    1200,
    1500,
    1800,
    2200,
    2700,
    3000,
    3500,
    4000,
    5000,
    6000,
    7000,
    8500,
    10000,
    13000,
    17000,
    22000,
    30000,
    40000,
]
RH = [1, 2, 3, 4, 6, 8, 12, 16]
out = []
for rho in RH:
    E.r = rho
    for n in NS:
        best = None
        for K1, K2, Q in E.candidates(n, Bp):
            d = per_sample(
                n,
                K1,
                K2,
                Q,
                r=rho,
                T=1500,
                seed=(n + K2 if rho == 2 else n + K2 + 100000 * rho),
            )
            P1, P2 = E.P1P2(n, K1, K2, Q)
            c = dict(
                K1=K1, K2=K2, Q=Q, L=float(d["Lget"]), pfail=float(P1 + (1 - P1) * P2)
            )
            if best is None or c["L"] < best["L"]:
                best = c
        out.append(dict(rho=rho, n=n, best=best))
        print(rho, n, best, flush=True)
        json.dump(out, open("k64_ratio.json", "w"), indent=1)
