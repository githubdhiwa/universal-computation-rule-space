# Is Universal Computation Easy to Find?

A network view of elementary cellular automaton rule space. Network Science course project by Dhiwakaran (22PT21) and Ranjana (22PT25).

Motivated by Akhlaghpour (2022), who argues that simple universal rule systems are easy for evolution to stumble on, we treat the 256 elementary cellular automata as a mutation network (node = rule, edge = one bit of the rule table flipped; an 8-dimensional hypercube) and ask where computation-capable rules sit.

## Findings

- Behaviour clusters in rule space: assortativity by class 0.18 (≈8 s.d. above a shuffled-label null).
- Complex rules are rare (4 of 256: Rule 110 and its symmetric twins) and isolated, but never more than 5 mutations away, as the hypercube predicts.
- Complex rules sit on the edge of chaos: 37.5% of their neighbours are chaotic vs 12.5% overall (p ≈ 0.0002).
- Switched on from most to least irregular, rules percolate abruptly: the largest cluster jumps from 13 to 33 rules at 16% occupation.
- Application: an edge-of-chaos search cuts the median number of rules tested to find a complex rule from 41 to 23.

## Contents

| Folder | What |
| --- | --- |
| `paper/` | Paper (`paper.pdf`) and LaTeX source with figures |
| `slides/` | Presentation (`.pptx` with speaker notes, `.pdf`) |
| `code/` | Python pipeline that regenerates every number, table and figure |

## Reproduce

```
cd code
pip install -r requirements.txt
python run_all.py   # ~1-2 minutes; outputs CSV/JSON results and figures/
```

| Paper section | Script |
| --- | --- |
| Data, classification | `eca_classify.py`, `eca_network.py` |
| RQ1-RQ2: clustering, distances, edge of chaos | `eca_network.py` |
| RQ3: percolation | `percolation.py` |
| RQ4: robustness | `robustness_checks.py` |
| Application: guided search | `search_application.py` |
| Figures | `paper_figures.py` |
