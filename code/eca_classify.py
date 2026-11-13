"""
Step 1: simulate and classify all 256 elementary cellular automata (ECA).

Measures per rule (averaged over several random initial conditions):
  - final_uniform : configuration becomes all-0 or all-1        -> Class 1 signal
  - period        : spacetime becomes periodic (up to a shift)   -> Class 2 signal
  - transient     : steps before that periodicity sets in        -> long = complex signal
  - compress      : zlib compression ratio of late spacetime,
                    normalised so random noise ~ 1               -> Class 3 signal
  - damage        : fraction of cells differing after flipping
                    one initial bit (damage spreading)           -> chaos signal
Outputs eca_features.csv
"""
import zlib
import numpy as np
import pandas as pd

N = 200          # cells (periodic boundary)
T = 1200         # time steps
SEEDS = 5
MAX_PERIOD = 300
RNG = np.random.default_rng(42)


def step(x, rule_bits):
    l, r = np.roll(x, 1), np.roll(x, -1)
    return rule_bits[(l << 2) | (x << 1) | r]


def run(rule, x0, T):
    bits = np.array([(rule >> i) & 1 for i in range(8)], dtype=np.uint8)
    out = np.empty((T + 1, len(x0)), dtype=np.uint8)
    out[0] = x0
    for t in range(T):
        out[t + 1] = step(out[t], bits)
    return out


def compress_ratio(block):
    raw = np.packbits(block).tobytes()
    return len(zlib.compress(raw, 9)) / len(raw)


# normaliser: compression ratio of true random bits of the same shape
_rand = RNG.integers(0, 2, size=(400, N), dtype=np.uint8)
RAND_RATIO = compress_ratio(_rand)


def periodicity(st):
    """Return (transient, period) where row t equals some shifted copy of row t-p.
    Detects first t at which the configuration repeats (up to spatial shift)."""
    # |FFT|^2 is shift-invariant -> cheap candidate key; verify shifts explicitly
    spec = np.abs(np.fft.rfft(st.astype(float), axis=1)) ** 2
    keys = np.round(spec, 3)
    seen = {}
    for t in range(len(st)):
        k = keys[t].tobytes()
        if k in seen:
            for t_prev in seen[k]:
                prev = st[t_prev]
                for s in range(st.shape[1]):
                    if np.array_equal(np.roll(prev, s), st[t]):
                        return t_prev, t - t_prev
            seen[k].append(t)
        else:
            seen[k] = [t]
    return len(st), 0  # no repeat found


def features(rule):
    fu, tr, per, cr, dm = [], [], [], [], []
    for s in range(SEEDS):
        x0 = RNG.integers(0, 2, size=N, dtype=np.uint8)
        st = run(rule, x0, T)
        last = st[-1]
        fu.append(float(last.min() == last.max()))
        t0, p = periodicity(st)
        tr.append(t0)
        per.append(p)
        cr.append(compress_ratio(st[-400:]) / RAND_RATIO)
        # damage spreading
        x1 = x0.copy()
        x1[N // 2] ^= 1
        st1 = run(rule, x1, 150)
        dm.append(float(np.mean(st1[-1] != st[150])))
    per_arr = np.array(per)
    return dict(
        rule=rule,
        uniform=np.mean(fu),
        transient=float(np.median(tr)),
        periodic_frac=float(np.mean(per_arr > 0)),
        period=float(np.median(per_arr[per_arr > 0])) if (per_arr > 0).any() else 0.0,
        compress=float(np.mean(cr)),
        damage=float(np.mean(dm)),
        lam=bin(rule).count("1") / 8.0,
    )


if __name__ == "__main__":
    rows = [features(r) for r in range(256)]
    df = pd.DataFrame(rows)
    df.to_csv("eca_features.csv", index=False)
    print("random ratio", RAND_RATIO)
    print(df.loc[[0, 4, 30, 45, 54, 90, 110, 150, 184, 232]].to_string())
