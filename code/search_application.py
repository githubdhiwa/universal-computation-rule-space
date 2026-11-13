"""
Application: finding a rare, computationally capable rule with as few tests as possible.

Problem. A designer searches a space of candidate rules (here the 256 ECA) for one with
complex, computation-capable behaviour. Testing a candidate is the expensive step
(here: simulating it and classifying it), so cost = number of DISTINCT rules tested.

Strategies compared (each run starts from a uniformly random rule):
  random      : test rules in random order (no network used)
  lambda      : test only rules with Langton lambda in {3/8, 4/8, 5/8}, in random order
                (uses the 'middle rows' observation, no network)
  blind_walk  : mutation random walk on the hypercube; every visited rule is tested
  edge_chaos  : mutation walk that uses the edge-of-chaos result. Walk blindly until a
                chaotic rule is found; then test ALL 8 neighbours of each chaotic rule
                reached, breadth-first (blind walk again if that region is used up)
  combined    : lambda sampling until a chaotic rule is found, then edge_chaos expansion
Ground truth labels come from eca_classes.csv (cls_auto). 20,000 runs per strategy.
"""
import json
import numpy as np
import pandas as pd

RUNS = 20000
lbl = pd.read_csv("eca_classes.csv", index_col=0)["cls_auto"].values.astype(int)
LAM = np.array([bin(r).count("1") for r in range(256)])
COMPLEX, CHAOTIC = 4, 3
rng = np.random.default_rng(2026)


mid = np.where((LAM >= 3) & (LAM <= 5))[0]


def nbrs(r):
    return [r ^ (1 << k) for k in range(8)]


def random_order(pool):
    order = rng.permutation(pool)
    hits = np.where(lbl[order] == COMPLEX)[0]
    return int(hits[0]) + 1


def blind_walk():
    r = int(rng.integers(256)); seen = {r}
    while lbl[r] != COMPLEX:
        r = nbrs(r)[rng.integers(8)]
        seen.add(r)
    return len(seen)


def edge_chaos():
    """Blind walk until a chaotic rule is found; then expand the chaotic region
    breadth-first, testing all 8 neighbours of every chaotic rule reached. If that
    region is used up without a complex rule, resume the blind walk from where we are."""
    r = int(rng.integers(256)); seen = {r}; expanded = set()
    while True:
        if lbl[r] == COMPLEX:
            return len(seen)
        if lbl[r] == CHAOTIC and r not in expanded:
            queue = [r]
            while queue:
                c = queue.pop(0)
                if c in expanded:
                    continue
                expanded.add(c)
                nb = nbrs(c); rng.shuffle(nb)
                for n in nb:
                    if n not in seen:
                        seen.add(n)
                        if lbl[n] == COMPLEX:
                            return len(seen)
                        if lbl[n] == CHAOTIC:
                            queue.append(n)
                r = c
        r = nbrs(r)[rng.integers(8)]            # blind step
        seen.add(r)


def combined():
    """Use both findings: sample untested middle-lambda rules until a chaotic rule turns up,
    then expand the chaotic region as in edge_chaos; repeat until a complex rule is found."""
    pool = list(rng.permutation(mid)); seen = set(); expanded = set()
    while True:
        r = None
        while pool:
            c = int(pool.pop())
            if c in seen:
                continue
            seen.add(c)
            if lbl[c] == COMPLEX:
                return len(seen)
            if lbl[c] == CHAOTIC:
                r = c; break
        if r is None:                       # middle rows exhausted: fall back to the rest
            pool = [int(x) for x in rng.permutation(256) if x not in seen]; continue
        queue = [r]
        while queue:
            c = queue.pop(0)
            if c in expanded:
                continue
            expanded.add(c)
            nb = nbrs(c); rng.shuffle(nb)
            for n in nb:
                if n not in seen:
                    seen.add(n)
                    if lbl[n] == COMPLEX:
                        return len(seen)
                    if lbl[n] == CHAOTIC:
                        queue.append(n)


def summary(x):
    x = np.asarray(x)
    return dict(mean=float(x.mean()), median=float(np.median(x)), p90=float(np.percentile(x, 90)),
                sem=float(x.std(ddof=1) / np.sqrt(len(x))), within_20=float((x <= 20).mean()))


raw = {
    "random": [random_order(np.arange(256)) for _ in range(RUNS)],
    "lambda": [random_order(mid) for _ in range(RUNS)],
    "blind_walk": [blind_walk() for _ in range(RUNS)],
    "edge_chaos": [edge_chaos() for _ in range(RUNS)],
    "combined": [combined() for _ in range(RUNS)],
}
np.savez("search_application_raw.npz", **{k: np.array(v) for k, v in raw.items()})
res = {k: summary(v) for k, v in raw.items()}
res.update({
    "n_lambda_pool": int(len(mid)),
    "complex_lambdas": [int(LAM[r]) for r in np.where(lbl == COMPLEX)[0]],
})
json.dump(res, open("search_application.json", "w"), indent=1)
for k, v in res.items():
    print(k, v)
