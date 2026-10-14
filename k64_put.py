# (O5) PUT time with coding included, Table VIII: 99th percentile of the time to serve a column and the
# fraction of columns served within the 4 s deadline, over 300 random networks per design and size.
import json, math, numpy as np
from eval_sim import kappa

MAC = 63.9e-6  # ms per multiply-add


def put_time(rng, Np, Nb, K1, K2, Q, sigma_MiB=32, U=25e6, U_pub=1e9):
    c_bits = sigma_MiB * 2**20 * 8
    G = rng.multinomial(Np, [1 / (K1 * K2)] * (K1 * K2)).reshape(K1, K2)
    B = rng.multinomial(Nb, [1 / (K1 * K2)] * (K1 * K2)).reshape(K1, K2)
    c = rng.integers(K2)
    cell_g = G[0, c]
    cell_b = B[0, c]
    if cell_g == 0:
        return math.inf
    order = np.zeros(cell_g + cell_b, bool)
    order[:cell_g] = True
    rng.shuffle(order)
    t_rel = [(j + 1) * c_bits / U_pub * 1e3 for j in np.nonzero(order)[0]]
    ncol = int(G[:, c].sum() + B[:, c].sum())
    ngood = int(G[:, c].sum())
    piece_ms = Q / kappa * (c_bits / K2) / U * 1e3
    elems = (sigma_MiB * 2**20 / K2) / 32 / kappa  # field elements per piece
    code_first = (
        0.0 if Q >= kappa else Q * elems * kappa * MAC
    )  # coding of the first destination's Q pieces; the rest overlaps the upload
    done = np.full(ngood, math.inf)
    for tr in t_rel:
        pos = rng.permutation(ncol)[:ngood]
        done = np.minimum(done, tr + code_first + (pos + 1) * piece_ms + 60.0)
    return float(done.max())


# (N+, K1, K2, Q of Algorithm 1 at 25 Mbit/s); N = 2 N+
cases = [(2000, 12, 64, 14), (6000, 11, 128, 20), (20000, 11, 128, 6)]
out = []
for Np, K1, K2, Q in cases:
    row = dict(Np=Np, K1=K1, K2=K2)
    for name, q in [("tcda", Q), ("rda", 64)]:
        rng = np.random.default_rng(Np * 3 + q)
        pt = [put_time(rng, Np, Np, K1, K2, q) for _ in range(300)]
        pt = [x for x in pt if x < math.inf]
        row[name] = dict(
            Q=q,
            p99=float(np.percentile(pt, 99)) / 1000,
            ok=float(np.mean(np.array(pt) <= 4000)),
        )
    out.append(row)
    print(row, flush=True)
json.dump(out, open("k64_put.json", "w"), indent=1)
