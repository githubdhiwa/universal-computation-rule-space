"""Percolation on CA rule space (a phase transition).
Switch on the top fraction f of rules ranked by irregularity (compression score:
chaotic rules first, then the complex ones, then the most irregular periodic rules),
and measure the largest connected cluster in the mutation network (8-cube).
Compare with keeping a random fraction f (classic site percolation)."""
import numpy as np, pandas as pd, networkx as nx
df = pd.read_csv("eca_features.csv")
score = df["compress"].values
G = nx.hypercube_graph(8)
G = nx.relabel_nodes(G, {n: int("".join(map(str, n)), 2) for n in G})
fs = np.linspace(0.02, 1, 50)

def giant(nodes):
    H = G.subgraph(nodes)
    return max((len(c) for c in nx.connected_components(H)), default=0) / 256

order = np.argsort(-score + 1e-9 * np.random.default_rng(0).random(256))
S_cx = [giant(order[: int(round(f * 256))]) for f in fs]
rng = np.random.default_rng(1)
S_rand = np.array([[giant(rng.permutation(256)[: int(round(f * 256))]) for f in fs] for _ in range(200)])
out = pd.DataFrame({"f": fs, "complexity_order": S_cx,
                    "random_mean": S_rand.mean(0), "random_sd": S_rand.std(0)})
out.to_csv("percolation.csv", index=False)
print(out.round(3).to_string())
