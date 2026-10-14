# Unified protocol-level simulator for the Numerical Evaluation.
# Network: N+ good and (r-1)N+ bad nodes placed uniformly on a K1 x K2 grid.
# Worst placement (Def. 1 / Lemma 2): publisher in row 0, client in row 1, target column drawn uniformly.
# Adversary: bad relays drop the chunk (or send wrongly-coded pieces, rejected by commitment check);
# bad collectors withhold; bad column peers stay silent (or send pieces that fail verification).
# Mechanism (PUT protocol): chunk enters column c iff a good node of cell (0,c) relays it; then every good node
# of column c holds Q designated pieces. The client queries all nodes of cell (1,c); each good collector
# doubles over a random order of its column peers, asking each min(Q, kappa-k) coordinates; designated
# vectors are innovative w.h.p. (validated with real field algebra).
import numpy as np, math

kappa = 64
pi = 48 / 2048


def collector(rng, n_good_peers, n_peers, Q):
    """Returns (rounds t, peers queried, coordinates received from good peers, success)."""
    order = np.zeros(n_peers, bool)
    order[:n_good_peers] = True
    rng.shuffle(order)
    k = Q
    pos = 0
    t = 0
    coords = 0
    while k < kappa and pos < n_peers:
        t += 1
        b = 2 ** (t - 1)
        batch = order[pos : pos + b]
        pos += len(batch)
        ask = min(Q, kappa - k)
        g = int(batch.sum())
        coords += g * ask
        k = min(kappa, k + g * ask)
    return t, pos, coords, k >= kappa


def attempt(rng, G, Bcol, c, Q):
    """One GET attempt at column c. G: good counts per cell (K1 x K2); Bcol: bad count per column.
    Returns dict(kind in {'ok','cell','short'}, rounds, queries(list), traffic)."""
    a = G[0, c]
    b = G[1, c]
    n = int(G[:, c].sum())
    Nc = n + int(Bcol[c])
    if b == 0:
        return dict(kind="cell", rounds=1, queries=[], traffic=0.0)
    if Q == kappa:
        if a == 0:
            return dict(kind="cell", rounds=1, queries=[], traffic=0.0)
        return dict(kind="ok", rounds=1, queries=[0] * b, traffic=b * (1 + pi))
    if (
        a == 0
    ):  # chunk never entered column: good collectors hold nothing and reply "absent"
        return dict(kind="cell", rounds=1, queries=[], traffic=0.0)
    ts = []
    qs = []
    tr = 0.0
    ok = False
    for _ in range(b):
        t, q, coords, s = collector(rng, n - 1, Nc - 1, Q)
        ts.append(t)
        qs.append(q)
        tr += coords * (1 / kappa + pi) + ((1 + pi) if s else 0.0)
        ok |= s
    if ok:
        return dict(kind="ok", rounds=1 + min(ts), queries=qs, traffic=tr)
    return dict(kind="short", rounds=1 + max(ts), queries=qs, traffic=tr)


def network(rng, Np, K1, K2, r):
    G = rng.multinomial(Np, [1 / (K1 * K2)] * (K1 * K2)).reshape(K1, K2)
    Bcol = (
        rng.multinomial(int(round((r - 1) * Np)), [1 / K2] * K2)
        if r > 1
        else np.zeros(K2, int)
    )
    return G, Bcol


def first_attempt_stats(Np, K1, K2, Q, r=2, T=20000, seed=1):
    """Empirical p_fail and its split (fresh network per trial, one attempt)."""
    rng = np.random.default_rng(seed)
    cnt = {"ok": 0, "cell": 0, "short": 0}
    for _ in range(T):
        G, B = network(rng, Np, K1, K2, r)
        c = rng.integers(K2)
        cnt[attempt(rng, G, B, c, Q)["kind"]] += 1
    return {k: v / T for k, v in cnt.items()}


def per_sample(Np, K1, K2, Q, r=2, T=4000, seed=2, cap=200):
    """Fresh network per sample; attempts at fresh random columns on that network until success.
    Returns means of: rounds per successful attempt, queries per collector, traffic per successful attempt,
    rounds per sample, traffic per sample, first-attempt failure split."""
    rng = np.random.default_rng(seed)
    ra = []
    qa = []
    fa = []
    rs = []
    fs = []
    first = {"ok": 0, "cell": 0, "short": 0}
    ncol = 0
    for _ in range(T):
        G, B = network(rng, Np, K1, K2, r)
        R = 0
        F = 0.0
        for i in range(cap):
            c = rng.integers(K2)
            o = attempt(rng, G, B, c, Q)
            if i == 0:
                first[o["kind"]] += 1
            R += o["rounds"]
            F += o["traffic"]
            if o["kind"] == "ok":
                ra.append(o["rounds"])
                qa.extend(o["queries"])
                fa.append(o["traffic"])
                ncol += len(o["queries"])
                break
        rs.append(R)
        fs.append(F)
    return dict(
        Latt=np.mean(ra),
        Gam=np.mean(qa) if qa else 0.0,
        Fatt=np.mean(fa),
        Lget=np.mean(rs),
        Fget=np.mean(fs),
        Fcol=sum(fa) / max(ncol, 1),
        first={k: v / T for k, v in first.items()},
    )
