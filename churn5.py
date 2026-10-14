# Outage under churn: RDA's schedule (churn4.py), plus one mass departure and recovery.
# After warm-up an object is published and followed. Steady churn: L oldest leave, L join per step.
# At age A0 lifetimes a fraction f of the good nodes leaves at once (uniformly at random).
# Recovery: churn goes on, with 2L joins per step until N_H good nodes are back.
# A joining node holds the object only if its column has h_Q holders (JOIN); otherwise the column has lost it.
import numpy as np, math, json, sys


def run(K1, K2, kappa, Qs, NH, f, L=50, A0=5, A1=5, seed=1):
    rng = np.random.default_rng(seed)
    life = NH // L
    T0 = A0 * life
    T1 = A1 * life
    maxn = 20 + NH + 3 * L * (T0 + T1 + 5)
    row = rng.integers(0, K1, maxn)
    col = rng.integers(0, K2, maxn)
    alive = np.zeros(maxn, bool)
    alive[:NH] = True
    nxt = NH  # network already at N_H (warm-up as in churn4)
    h = {Q: math.ceil(kappa / Q) for Q in Qs}

    def cellof(mask):
        c = np.zeros((K1, K2), int)
        np.add.at(c, (row[mask], col[mask]), 1)
        return c

    rp = rng.integers(K1)
    entered = cellof(alive)[rp] > 0
    hold = {Q: alive & entered[col] for Q in Qs}
    rows_other = [r for r in range(K1) if r != rp]
    out = {Q: [] for Q in Qs}
    size = []
    for t in range(1, T0 + T1 + 1):
        if t == T0 + 1:  # outage
            idx = np.flatnonzero(alive)
            gone = rng.choice(idx, int(round(f * len(idx))), replace=False)
            alive[gone] = False
            for Q in Qs:
                hold[Q][gone] = False
        idx = np.flatnonzero(alive)
        lv = idx[:L]
        alive[lv] = False  # oldest leave
        nj = 2 * L if alive.sum() + L < NH else L
        nj = min(nj, NH - int(alive.sum())) if alive.sum() + nj > NH else nj
        new = np.arange(nxt, nxt + nj)
        nxt += nj
        alive[new] = True
        occ = cellof(alive)[rows_other] > 0
        for Q in Qs:
            hold[Q][lv] = False
            cnt = np.bincount(col[hold[Q]], minlength=K2)
            ok = cnt[col[new]] >= h[Q]
            hold[Q][new[ok]] = True
            dec = np.bincount(col[hold[Q]], minlength=K2) >= h[Q]
            out[Q].append((float(1 - (occ & dec[None, :]).mean()), int((~dec).sum())))
        size.append(int(alive.sum()))
    return out, size, life, T0


if __name__ == "__main__":
    K1, K2, kappa = 12, 64, 64
    NH = int(sys.argv[1])
    Qs = [int(x) for x in sys.argv[2].split(",")]
    res = []
    for f in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
        acc = {Q: dict(before=[], trough=[], after=[], lost=[]) for Q in Qs}
        for seed in range(1, 11):
            out, size, life, T0 = run(K1, K2, kappa, Qs, NH, f, seed=seed)
            for Q in Qs:
                fr = [x[0] for x in out[Q]]
                acc[Q]["before"].append(np.mean(fr[T0 - life : T0]))
                acc[Q]["trough"].append(fr[T0])
                acc[Q]["after"].append(np.mean(fr[-life:]))
                acc[Q]["lost"].append(out[Q][-1][1])
        row = dict(
            f=f,
            **{str(Q): {k: float(np.mean(v)) for k, v in acc[Q].items()} for Q in Qs},
        )
        res.append(row)
        print(
            f,
            {
                Q: {k: round(float(np.mean(v)), 3) for k, v in acc[Q].items()}
                for Q in Qs
            },
            flush=True,
        )
    json.dump(dict(NH=NH, Qs=Qs, res=res), open(f"churn5_{NH}.json", "w"), indent=1)
