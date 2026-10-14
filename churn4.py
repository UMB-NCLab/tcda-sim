# RDA-style churn (RDA Sec. 6.2) with the publisher's contact cell included (worst placement: publisher and client in different rows).
import numpy as np, math, json, sys
from exact import pchunk


def pairfail(cell, cnt_ok):
    # mean over ordered row pairs (rp != r) and columns c of 1[cell(rp,c)=0 or cell(r,c)=0 or column not decodable]
    K1, K2 = cell.shape
    occ = (cell > 0).astype(float)  # K1 x K2
    s = occ.sum(0)  # occupied cells per column
    both = s * s - s  # ordered pairs of distinct occupied rows
    ok = both * cnt_ok
    return 1 - ok.sum() / (K1 * (K1 - 1) * K2)


def run(K1, K2, kappa, Qs, NH=2500, L=50, T_after=3600, seed=1):
    rng = np.random.default_rng(seed)
    maxn = 20 + NH + L * (T_after + 5)
    row = rng.integers(0, K1, maxn)
    col = rng.integers(0, K2, maxn)
    alive = np.zeros(maxn, bool)
    alive[:20] = True
    nxt = 20
    head = 0
    h = {Q: math.ceil(kappa / Q) for Q in Qs}
    grow = []

    def cellof(mask):
        c = np.zeros((K1, K2), int)
        np.add.at(c, (row[mask], col[mask]), 1)
        return c

    while True:
        n = int(alive.sum())
        cell = cellof(alive)
        cnt = np.bincount(col[alive], minlength=K2)
        grow.append((n, {Q: pairfail(cell, (cnt >= h[Q]).astype(float)) for Q in Qs}))
        if n >= NH:
            break
        alive[nxt] = True
        nxt += 1
    # object published now by a publisher in row rp; column c receives it iff cell (rp,c) has an honest party
    rp = rng.integers(K1)
    cell = cellof(alive)
    entered = cell[rp] > 0
    hold = {Q: alive & entered[col] for Q in Qs}
    per = {Q: [] for Q in Qs}
    rows_other = [r for r in range(K1) if r != rp]
    for a in range(1, T_after + 1):
        lv = np.arange(head, head + L)
        head += L
        alive[lv] = False
        new = np.arange(nxt, nxt + L)
        nxt += L
        alive[new] = True
        cell = cellof(alive)
        occ = cell[rows_other] > 0
        for Q in Qs:
            hold[Q][lv] = False
            cnt = np.bincount(col[hold[Q]], minlength=K2)
            ok = cnt[col[new]] >= h[Q]
            hold[Q][new[ok]] = True
            cnt = np.bincount(col[hold[Q]], minlength=K2)
            dec = cnt >= h[Q]
            fail = 1 - (occ & dec[None, :]).mean()
            per[Q].append((a, float(fail), int((~dec).sum())))
    return grow, per


if __name__ == "__main__":
    K1, K2, kappa = 12, 64, 64
    # usage: python3 churn4.py NH Q1,Q2,...  [lifetimes]
    # Without the third argument the run lasts 3600 steps after warm-up (72 lifetimes at NH=2500, since a
    # lifetime is NH/50 steps). With it, the run lasts that many lifetimes and the file name ends in _<lifetimes>lt.
    NH = int(sys.argv[1])
    Qs = [int(x) for x in sys.argv[2].split(",")]
    LT = int(sys.argv[3]) if len(sys.argv) > 3 else None
    T = 3600 if LT is None else LT * NH // 50
    out = dict(
        NH=NH,
        Qs=Qs,
        steps=T,
        exact={str(q): float(pchunk(NH, K1, K2, q)) for q in Qs},
        seeds=[],
    )
    for seed in range(1, 6):
        grow, per = run(K1, K2, kappa, Qs, NH=NH, T_after=T, seed=seed)
        out["seeds"].append(
            dict(
                grow=[(n, {str(q): v for q, v in m.items()}) for n, m in grow],
                per={str(q): v for q, v in per.items()},
            )
        )
        print(NH, seed, "done", flush=True)
    json.dump(
        out,
        open(f"churn4_{NH}.json" if LT is None else f"churn4_{NH}_{LT}lt.json", "w"),
    )
