# TCDA simulation code

Code and results behind the numbers, tables and figures of the paper
"TCDA: Tunable Coded Distributed Arrays for Data Availability Sampling".

Requirements: Python 3.10 or later with `numpy` and `scipy`; a C compiler for the one benchmark.

```
pip install numpy scipy
```

Each script writes its JSON output to the current directory. The outputs used in the paper are in `results/`.
Seeds are fixed in every script, so a rerun reproduces the stored results exactly.

## Which script produces what

| Paper | Command | Output |
|---|---|---|
| Table II (bootstrap sizes), Table III (TCDA vs RDA), worked examples in the text | `python3 ex_k64.py` | printed |
| Fig. 7, (O1) growth of the GET cost with bad nodes | `python3 k64_growth.py` | `k64_growth.json` |
| Table VI and Fig. 10, feasible ranges of good nodes | `DMAX=1/10 python3 eval_d10.py`, then `DMAX=1/6` | `k64_ranges_d1_10.json`, `k64_ranges_d1_6.json` |
| Table VI and Fig. 8, (O2) design chosen by Algorithm 1, its latency and failure probability; check of the closed-form planner | `DMAX=1/10 python3 eval_d10_alg.py`, then `DMAX=1/6` | `k64_alg_d1_10.json`, `k64_alg_d1_6.json` |
| Fig. 9, (O3) latency over the good nodes and the ratio N_max/N+ | `DMAX=1/10 python3 k64_ratio.py` | `k64_ratio.json` |
| Fig. 11, (O4) Sybil attack on PeerDAS | `python3 k64_sybil.py` | `k64_sybil.json` |
| (O4) targeted attack on one column | `python3 k64_target.py` | `k64_target.json` |
| Table VIII, (O5) PUT time with coding included | `python3 k64_put.py` | `k64_put.json` |
| Coding speed used by Table VIII (64 ns per multiply–add) | `gcc -O3 -march=native bench/frmac.c -o frmac && ./frmac` | printed |
| Table IX, (O6) stored object under steady churn | see "Churn" below | `churn4_*.json` |
| Fig. 12, (O6) mass departure of good nodes | `python3 churn5.py 2500 64,12,2` | `churn5_2500.json` |

Table VII (overall comparison at 2000 good nodes) is assembled from these outputs: the design and its round trips from
`k64_alg_d1_10.json`, the failure probabilities from `k64_sybil.json`, and the share of columns served in time from `k64_put.json`.

### Churn (Table IX)

```
python3 churn4.py 1600 64,17,8,4
python3 churn4.py 2500 64,12,4,2
python3 churn4.py 4000 64,7,2 72
python3 churn4_sum.py
```

`churn4.py NH Qs [lifetimes]` follows RDA's schedule (its Section 6.2): 20 initial nodes, one joins per step until
`NH` are present, then 50 leave (oldest first) and 50 join at every step. A node therefore stays `NH/50` steps, one lifetime.
Only good nodes are simulated, as in RDA. A node that joins holds the stored object only if its column still has
`h_Q = ceil(64/Q)` holders to sync from; a column with fewer holders has lost the object.

Without the third argument a run lasts 3600 steps, which is 72 lifetimes at `NH = 2500` and 112.5 at `NH = 1600`.
At `NH = 4000` that would be only 45 lifetimes, so that run takes the third argument `72`.
`churn4_sum.py` reads the first 72 lifetimes of each run and prints the rows of Table IX, together with the time at
which the failure rate first exceeds delta.

`churn5.py NH Qs` adds one outage: after 5 lifetimes a fraction of the good nodes, from 0 to 90%, leaves at once,
and the network refills at a net 50 nodes per step. It reports the failure rate of the stored object right after
the outage and after recovery, as a mean of 10 seeds. The paper uses `NH = 2500`; the results for 1600 and 4000 are included.

## Core modules

- `newsetting.py`: the parameters: kappa = 64, M = 128, delta = 1/4, symbol size, budgets.
- `exact.py`: the exact failure probability `pchunk` (Corollary 1), the bootstrap `nmin` = N+_delta(Q) (Theorem 1), and `qdelta` = Q_delta.
- `eval_sim.py`: the protocol-level simulator: uniform placement on the grid, the worst-case adversary, the collector's doubling GET, and the statistics per attempt and per sample (failed attempts included).
- `eval_d10.py`: Algorithm 1: candidate grids under the degree limit, the 99th-percentile column, Q_max, and the feasibility test. The degree limit is read from the environment variable `DMAX` (default `1/10`).
- `eval_attack.py`: the PeerDAS failure model (closed form and Monte Carlo) and the grid simulator with bad nodes placed in a chosen column.

## Parameters

| Parameter | Value | Where |
|---|---|---|
| kappa, M, delta | 64, 128, 1/4 | `newsetting.py`; `eval_d10.py` (`M`, `DELTA`) |
| Object size | 32 MiB | `eval_d10.py`, `k64_put.py` |
| Uplinks | 25, 50, 100 Mbit/s | loops in `eval_d10.py`, `eval_d10_alg.py` |
| PUT budget | 4 s at the uplink rate, minus the publisher's 0.87 s | `eval_d10.py` (`TPUB`) |
| Column percentile for the PUT | 99th | `eval_d10.py` |
| Storage budget per node, retention | 32 GiB (half of PeerDAS's 0.5 MiB per object), 131072 blocks | `eval_d10.py` (`Bs`, overridable with `BS_MIB`; `W`) |
| Degree limit | 1/10, also 1/6 | environment variable `DMAX` |
| Churn grid | (12, 64) | `churn4.py`, `churn5.py` |

If you change a parameter that appears in more than one file, change it in all of them.

## Running times (2 cores)

| Script | Time |
|---|---|
| `ex_k64.py`, `k64_sybil.py`, `k64_put.py`, `churn4_sum.py` | seconds |
| `k64_growth.py` | 20 s |
| `k64_target.py`, `churn5.py 2500 ...`, each `churn4.py` run | about 30 s |
| `eval_d10_alg.py` | 3 to 4 min per degree limit |
| `k64_ratio.py` | 4 min |
| `eval_d10.py` | 7 to 8 min per degree limit |

## Notes

- The coding speed in `k64_put.py` (`MAC`, 63.9 ns per multiply–add) was measured with `bench/frmac.c` on a virtual machine
  with 2 cores of a 2.8 GHz Intel Xeon. Rerun the benchmark on your own machine and update the constant if you report new timings.
- `k64_ratio.py` uses the same samples and seeds as `eval_d10_alg.py` at the ratio 2, so Fig. 9 agrees with Fig. 8 there.
- In `eval_sim.py` the ratio of proof size to symbol size (beta in the paper) is called `pi`.
