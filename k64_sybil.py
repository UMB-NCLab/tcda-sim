# (O4) Sybil attack, Fig. 11: per-sample failure probability at N+ = 2000 good nodes as Sybils join.
# PeerDAS: closed form under the paper's model (eval_attack.peerdas_fail_exact), C = 8 or 16 contacted advertisers.
# Grid designs: exact p_fail of the design of the overall comparison, (12,64), Q=14 (TCDA) and Q=64 (RDA);
# it depends on the good nodes only, so it is the same at every Sybil count.
import json
from eval_attack import peerdas_fail_exact
from exact import pchunk

Np = 2000
out = dict(
    Np=Np,
    rows=[],
    tcda=float(pchunk(Np, 12, 64, 14)),
    rda=float(pchunk(Np, 12, 64, 64)),
)
for ratio in [0, 0.1, 0.25, 0.5, 1, 2, 5, 10]:
    Nb = int(ratio * Np)
    row = dict(
        sybils_per_good=ratio,
        peerdas_C8=float(peerdas_fail_exact(Np, Nb, C=8)),
        peerdas_C16=float(peerdas_fail_exact(Np, Nb, C=16)),
    )
    out["rows"].append(row)
    print(row, flush=True)
# smallest Sybil count at which PeerDAS exceeds delta = 1/4
for C in (8, 16):
    S = next(s for s in range(0, 4 * Np, 5) if peerdas_fail_exact(Np, s, C=C) > 0.25)
    out[f"crossing_C{C}"] = S / Np
    print("C", C, "exceeds delta from", S / Np, "Sybils per good node")
print("grid designs: TCDA", round(out["tcda"], 4), "RDA", round(out["rda"], 4))
json.dump(out, open("k64_sybil.json", "w"), indent=1)
