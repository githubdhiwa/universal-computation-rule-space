"""
RQ4: are the network results robust to how rules are classified and simulated?

1. Classifier thresholds: compression cut-off for 'complex' in {0.3, 0.4, 0.5} x
   transient cut-off for 'periodic' in {400, 600, 800} steps.
2. Rule 54 counted as complex (cls_ref).
3. A fresh simulation: different random seed and a wider lattice (300 cells).
For each setting: number of complex rules, assortativity (vs null), complex-complex
edges, mean distance to nearest complex rule, chaotic share next to complex rules (vs null).
Also: how many chaotic clusters touch a complex rule (explains the search results).
"""
import json
import numpy as np
import pandas as pd
import networkx as nx
import eca_classify as ec

RNG = np.random.default_rng(99)
G = nx.Graph()
G.add_edges_from((r, r ^ (1 << k)) for r in range(256) for k in range(8))
E = np.array([(r, r ^ (1 << k)) for r in range(256) for k in range(8) if r < r ^ (1 << k)])
D = np.array([[bin(a ^ b).count("1") for b in range(256)] for a in range(256)])


def classify(df, comp_cut=0.4, trans_cut=600):
    lbl = np.full(256, 3)
    for r in range(256):
        f = df.loc[r]
        if f.uniform >= 0.6:
            lbl[r] = 1
        elif f.periodic_frac >= 0.6 and f.transient < trans_cut:
            lbl[r] = 2
        elif f.compress < comp_cut:
            lbl[r] = 4
    return lbl


def newman_r(lbl):
    M = np.zeros((4, 4))
    for a, b in E:
        M[lbl[a] - 1, lbl[b] - 1] += 1
        M[lbl[b] - 1, lbl[a] - 1] += 1
    e = M / M.sum(); a = e.sum(1)
    return (np.trace(e) - (a * a).sum()) / (1 - (a * a).sum())


def chaotic_share(lbl):
    cx = np.where(lbl == 4)[0]
    if len(cx) == 0:
        return np.nan
    return np.mean([lbl[r ^ (1 << k)] == 3 for r in cx for k in range(8)])


def metrics(lbl, nperm=2000):
    cx = np.where(lbl == 4)[0]
    r_obs = newman_r(lbl)
    nulls = [newman_r(RNG.permutation(lbl)) for _ in range(300)]
    ch = chaotic_share(lbl)
    ch_null = np.array([chaotic_share(RNG.permutation(lbl)) for _ in range(nperm)])
    sub = D[np.ix_(cx, cx)]
    return dict(
        n_complex=int(len(cx)),
        complex_rules=[int(x) for x in cx],
        assortativity=float(r_obs),
        assort_z=float((r_obs - np.mean(nulls)) / np.std(nulls)),
        complex_complex_edges=int(((sub == 1).sum()) // 2),
        mean_dist_to_complex=float(D[:, cx].min(1).mean()) if len(cx) else None,
        chaotic_share_next_to_complex=float(ch),
        chaotic_share_overall=float((lbl == 3).mean()),
        enrichment_p=float((ch_null >= ch).mean()),
    )


out = {"thresholds": []}
base = pd.read_csv("eca_features.csv").set_index("rule")
for cc in (0.3, 0.4, 0.5):
    for tc in (400, 600, 800):
        m = metrics(classify(base, cc, tc))
        m.update(compress_cut=cc, transient_cut=tc)
        out["thresholds"].append(m)

ref = classify(base)
ref[[54, 147]] = 4
out["rule54_complex"] = metrics(ref)

# fresh simulation: new seed, wider lattice
ec.N = 300
ec.RNG = np.random.default_rng(7)
ec.RAND_RATIO = ec.compress_ratio(ec.RNG.integers(0, 2, size=(400, ec.N), dtype=np.uint8))
fresh = pd.DataFrame([ec.features(r) for r in range(256)]).set_index("rule")
fresh_lbl = classify(fresh)
out["fresh_simulation"] = metrics(fresh_lbl)
out["fresh_simulation"]["agreement_with_main"] = float((fresh_lbl == classify(base)).mean())

# chaotic clusters and whether they touch a complex rule (main labels)
lbl = classify(base)
H = G.subgraph([r for r in range(256) if lbl[r] == 3])
comps = list(nx.connected_components(H))
touch = [any(lbl[n ^ (1 << k)] == 4 for n in c for k in range(8)) for c in comps]
out["chaotic_clusters"] = dict(n=len(comps), sizes=sorted((len(c) for c in comps), reverse=True),
                               touching_complex=int(sum(touch)),
                               rules_in_touching=int(sum(len(c) for c, t in zip(comps, touch) if t)))
json.dump(out, open("robustness_checks.json", "w"), indent=1)

print(pd.DataFrame(out["thresholds"])[["compress_cut", "transient_cut", "n_complex", "assortativity", "assort_z",
      "complex_complex_edges", "mean_dist_to_complex", "chaotic_share_next_to_complex", "enrichment_p"]].round(3).to_string())
for k in ("rule54_complex", "fresh_simulation", "chaotic_clusters"):
    print(k, out[k])
