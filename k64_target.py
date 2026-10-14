# (O4) targeted placement on the design of the overall comparison: (12,64), N+=2000, Q=14 (TCDA) and Q=64 (RDA).
import json
from eval_attack import fail_rate
from exact import pchunk

K1, K2 = 12, 64
n = 2000
Nb = 2000
out = []
for frac in [0, 0.25, 0.5, 1.0]:
    for Q in [14, 64]:
        p, l = fail_rate(
            n,
            K1,
            K2,
            Q,
            Nb,
            "designated",
            T=20000,
            seed=int(frac * 100) + Q,
            target=True,
            frac=frac,
        )
        out.append(
            dict(
                frac=frac,
                Q=Q,
                p=p,
                exact=float(pchunk(n, K1, K2, Q)),
                Latt=float(l),
                Ncol=n / K2 + Nb * (1 - frac) / K2 + Nb * frac,
            )
        )
        print(out[-1], flush=True)
json.dump(out, open("k64_target.json", "w"), indent=1)
