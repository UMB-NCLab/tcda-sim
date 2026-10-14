# (O6) Table X: a stored object under RDA's churn schedule, over the first 72 lifetimes after warm-up.
# Reads the outputs of churn4.py from results/ (or the current directory).
#   python3 churn4_sum.py
import json, os, numpy as np

D = 0.25
LT = 72
RUNS = [
    (1600, "churn4_1600.json", [64, 17, 4]),  # RDA, optimal Q, Q_delta
    (2500, "churn4_2500.json", [64, 12, 2]),
    (4000, "churn4_4000_72lt.json", [64, 7, 2]),
]


def load(f):
    for d in (".", "results"):
        if os.path.exists(os.path.join(d, f)):
            return json.load(open(os.path.join(d, f)))
    raise FileNotFoundError(f)


print(
    "N_H   design      h_Q  robust  worst  lost    first time above delta (lifetimes, per seed)"
)
for NH, f, Qs in RUNS:
    d = load(f)
    life = NH / 50
    n = int(LT * life)
    for q in Qs:
        R = []
        MX = []
        LOST = []
        FIRST = []
        for s in d["seeds"]:
            ser = s["per"][str(q)]
            assert len(ser) >= n, "run shorter than 72 lifetimes"
            p = np.array([x[1] for x in ser[:n]])
            R.append(np.mean(p <= D))
            MX.append(p.max())
            LOST.append(ser[n - 1][2])
            FIRST.append(
                round((int(np.argmax(p > D)) + 1) / life, 2) if (p > D).any() else None
            )
        name = "RDA" if q == 64 else f"TCDA/Q={q}"
        print(
            f"{NH:<5} {name:<11} {-(-64 // q):>3}  {np.mean(R) * 100:5.0f}%  {max(MX):5.2f}  {min(LOST)}-{max(LOST):<4}  {FIRST}"
        )
