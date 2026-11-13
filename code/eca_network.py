"""
Step 2: rule-space network of the 256 elementary CA.

Nodes  = rules 0..255
Edges  = rules whose 8-bit tables differ in exactly one bit (one "mutation")
         -> the 8-dimensional hypercube Q8 (256 nodes, 1024 edges, degree 8)

Questions:
  * Are behaviour classes clustered in rule space (assortativity)?
  * Do complex rules form a connected neutral network or isolated islands?
  * How many mutations from a random rule to the nearest complex rule?
  * Is the complexity landscape smooth (autocorrelation vs Hamming distance)?
All compared against null models that shuffle class labels over the same graph.
"""
import json
import numpy as np
import pandas as pd
import networkx as nx

RNG = np.random.default_rng(7)
NPERM = 5000

df = pd.read_csv("eca_features.csv").set_index("rule")

# ---------------------------------------------------------------- classes
def auto_class(r):
    f = df.loc[r]
    if f.uniform >= 0.6:
        return 1
    if f.periodic_frac >= 0.6 and f.transient < 600:
        return 2
    if f.compress < 0.4:          # long transient + structured  -> complex
        return 4
    return 3


df["cls_auto"] = [auto_class(r) for r in df.index]
df["cls_ref"] = df["cls_auto"]
df.loc[[54, 147], "cls_ref"] = 4  # textbook Class 4 adds Rule 54 family
df[["cls_auto", "cls_ref"]].to_csv("eca_classes.csv")

# ---------------------------------------------------------------- graph
G = nx.Graph()
G.add_nodes_from(range(256))
for r in range(256):
    for k in range(8):
        s = r ^ (1 << k)
        if s > r:
            G.add_edge(r, s, bit=k)
E = np.array(G.edges())
D = np.array([[bin(a ^ b).count("1") for b in range(256)] for a in range(256)])


def assort(lbl):
    same = (lbl[E[:, 0]] == lbl[E[:, 1]]).mean()
    return same


def mixing(lbl, classes=(1, 2, 3, 4)):
    M = np.zeros((4, 4))
    for r in range(256):
        for k in range(8):
            M[lbl[r] - 1, lbl[r ^ (1 << k)] - 1] += 1
    return M / M.sum(axis=1, keepdims=True)


def components(lbl, c):
    nodes = [r for r in range(256) if lbl[r] == c]
    H = G.subgraph(nodes)
    sizes = sorted((len(x) for x in nx.connected_components(H)), reverse=True)
    return sizes


def dist_to_class(lbl, c):
    targets = np.where(lbl == c)[0]
    return D[:, targets].min(axis=1)


def min_pairwise(lbl, c):
    t = np.where(lbl == c)[0]
    sub = D[np.ix_(t, t)].astype(float)
    np.fill_diagonal(sub, np.inf)
    return sub.min(axis=1)


results = {}
for name in ["cls_auto", "cls_ref"]:
    lbl = df[name].values.astype(int)
    out = {"counts": {int(c): int((lbl == c).sum()) for c in (1, 2, 3, 4)}}
    nx.set_node_attributes(G, {r: int(lbl[r]) for r in range(256)}, "cls")
    out["assortativity_coef"] = nx.attribute_assortativity_coefficient(G, "cls")
    obs_same = assort(lbl)
    null_same = np.array([assort(RNG.permutation(lbl)) for _ in range(NPERM)])
    null_coef = []
    for _ in range(500):
        p = RNG.permutation(lbl)
        nx.set_node_attributes(G, {r: int(p[r]) for r in range(256)}, "cls")
        null_coef.append(nx.attribute_assortativity_coefficient(G, "cls"))
    out["assort_null_mean"] = float(np.mean(null_coef))
    out["assort_null_sd"] = float(np.std(null_coef))
    out["same_class_edge_frac"] = float(obs_same)
    out["same_class_null_mean"] = float(null_same.mean())
    out["same_class_p"] = float((null_same >= obs_same).mean())
    out["mixing"] = mixing(lbl).round(3).tolist()

    # components per class, observed vs null largest-component size
    comp = {}
    for c in (1, 2, 3, 4):
        sizes = components(lbl, c)
        null_lcc = [components(RNG.permutation(lbl), c)[0] for _ in range(1000)]
        comp[int(c)] = dict(
            n_components=len(sizes), largest=sizes[0], sizes=sizes,
            null_largest_mean=float(np.mean(null_lcc)),
            p_largest_ge=float((np.array(null_lcc) >= sizes[0]).mean()),
        )
    out["components"] = comp

    # distance from every rule to nearest complex (class 4) rule
    d4 = dist_to_class(lbl, 4)
    nd = np.array([dist_to_class(RNG.permutation(lbl), 4).mean() for _ in range(NPERM)])
    out["dist_to_complex"] = dict(
        hist={int(k): int((d4 == k).sum()) for k in range(9)},
        mean=float(d4.mean()), null_mean=float(nd.mean()), null_sd=float(nd.std()),
        p_lower=float((nd <= d4.mean()).mean()),
        mean_from_simple=float(d4[(lbl == 1) | (lbl == 2)].mean()),
        mean_from_chaotic=float(d4[lbl == 3].mean()),
    )
    # how far apart are complex rules from each other?
    mp = min_pairwise(lbl, 4)
    nmp = np.array([min_pairwise(RNG.permutation(lbl), 4).mean() for _ in range(NPERM)])
    out["complex_nn_dist"] = dict(obs=mp.tolist(), mean=float(mp.mean()),
                                  null_mean=float(nmp.mean()),
                                  p_lower=float((nmp <= mp.mean()).mean()))
    # neighbours of complex rules
    neigh = {}
    for r in np.where(lbl == 4)[0]:
        neigh[int(r)] = [int(lbl[r ^ (1 << k)]) for k in range(8)]
    out["complex_neighbour_classes"] = neigh
    results[name] = out

# ------------------------------------------------- landscape autocorrelation
x = df["compress"].values
acf = {}
for d in range(1, 9):
    i, j = np.where(np.triu(D == d))
    acf[d] = float(np.corrcoef(x[i], x[j])[0, 1])
results["compress_autocorr"] = acf

# ------------------------------------------------- per-bit mutational sensitivity
lbl = df["cls_auto"].values.astype(int)
bit_change = {}
for k in range(8):
    bit_change[k] = float(np.mean([lbl[r] != lbl[r ^ (1 << k)] for r in range(256)]))
results["class_change_by_bit"] = bit_change

json.dump(results, open("eca_network_results.json", "w"), indent=1)

for name in ["cls_auto", "cls_ref"]:
    o = results[name]
    print("==", name, o["counts"])
    print(" assortativity", round(o["assortativity_coef"], 3), "null",
          round(o["assort_null_mean"], 3), "+-", round(o["assort_null_sd"], 3))
    print(" same-class edges", round(o["same_class_edge_frac"], 3), "null",
          round(o["same_class_null_mean"], 3), "p", o["same_class_p"])
    for c, v in o["components"].items():
        print(f"  class {c}: {v['n_components']} comps, largest {v['largest']} "
              f"(null {v['null_largest_mean']:.1f}, p={v['p_largest_ge']:.3f})")
    print(" dist to complex", o["dist_to_complex"])
    print(" complex nn dist", o["complex_nn_dist"])
    print(" neighbours of complex", o["complex_neighbour_classes"])
    print(" mixing", np.array(o["mixing"]))
print("autocorr", acf)
print("bit sensitivity", bit_change)

# ------------------------------------------------- edge-of-chaos enrichment test
# Are chaotic rules over-represented among the neighbours of complex rules?
rng2 = np.random.default_rng(1)
enrich = {}
for name in ["cls_auto", "cls_ref"]:
    lbl = df[name].values.astype(int)

    def chaotic_share(l):
        nb = [l[r ^ (1 << k)] for r in np.where(l == 4)[0] for k in range(8)]
        return float(np.mean(np.array(nb) == 3))

    obs = chaotic_share(lbl)
    null = np.array([chaotic_share(rng2.permutation(lbl)) for _ in range(20000)])
    enrich[name] = dict(obs=obs, null=float(null.mean()), p=float((null >= obs).mean()))
    print(f"chaotic share next to complex ({name}): {obs:.3f}  chance {null.mean():.3f}  p={enrich[name]['p']:.4f}")
results["complex_neighbour_chaotic_enrichment"] = enrich
json.dump(results, open("eca_network_results.json", "w"), indent=1)
