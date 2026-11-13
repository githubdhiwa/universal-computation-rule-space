"""Regenerate every number, table and figure in the paper (about 1-2 minutes)."""
import os
import runpy
import time

os.makedirs("figures", exist_ok=True)
STEPS = [
    ("Simulate and measure all 256 rules", "eca_classify.py"),
    ("Mutation network: classes, assortativity, distances, edge of chaos (RQ1-RQ2)", "eca_network.py"),
    ("Percolation (RQ3)", "percolation.py"),
    ("Robustness checks (RQ4)", "robustness_checks.py"),
    ("Application: searching for a complex rule (RQ5)", "search_application.py"),
    ("Paper figures", "paper_figures.py"),
]
for title, script in STEPS:
    t = time.time()
    print(f"\n=== {title} ({script}) ===")
    runpy.run_path(script, run_name="__main__")
    print(f"--- done in {time.time() - t:.0f}s")
