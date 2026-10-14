import json, math
from eval_d10 import *

Ns = [
    321,
    330,
    400,
    500,
    600,
    700,
    800,
    868,
    900,
    966,
    1000,
    1200,
    1500,
    1800,
    2000,
    2500,
    3000,
    3500,
    4000,
    5000,
    5660,
    6000,
    7000,
    7215,
    8000,
    10000,
    13000,
    20000,
    30000,
    40000,
]
out = []
for mb in [25, 50, 100]:
    Bp = mb * 1e6 * 4 / 8 / 2**20
    for n in Ns:
        cands = candidates(n, Bp)
        best = None
        bestb = None
        rda = None
        for K1, K2, Q in cands:
            d = per_sample(n, K1, K2, Q, r=r, T=1500, seed=n + K2)
            P1, P2 = P1P2(n, K1, K2, Q)
            c = dict(
                K1=K1,
                K2=K2,
                Q=Q,
                L=d["Lget"],
                Lb=Lsucc_bound(Q) + failterm(n, K1, K2, Q),
                put=Q / kappa * Nhat(n, K2) * sigma / K2,
                pfail=P1 + (1 - P1) * P2,
            )
            if best is None or c["L"] < best["L"]:
                best = c
            if bestb is None or c["Lb"] < bestb["Lb"]:
                bestb = c
            if Q == kappa and (rda is None or c["L"] < rda["L"]):
                rda = c
        out.append(dict(mb=mb, n=n, tcda=best, planner=bestb, rda=rda))
        print(
            mb,
            n,
            best,
            "| planner",
            None if bestb is None else (bestb["K1"], bestb["K2"], bestb["Q"]),
            "| rda",
            None if rda is None else (rda["K1"], rda["K2"]),
            flush=True,
        )
import os

TAG = os.environ.get("DMAX", "1/10").replace("/", "_")
json.dump(out, open("k64_alg_d" + TAG + ".json", "w"), indent=1, default=float)
