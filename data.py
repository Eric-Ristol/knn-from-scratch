"""
data.py -- regenerate trees.csv (a synthetic 2-feature, 2-class dataset).

This script is a utility, not part of the classifier. Run it once (or any time
you want a fresh dataset) and both knn.py and knn.c will read the resulting
trees.csv.

    python data.py                # 200 samples, seed=0 (default)
    python data.py 500            # 500 samples, seed=0
    python data.py 500 7          # 500 samples, seed=7

Output format (CSV, header row):

    height,width,label
    23.164248,0.329318,Pine
    27.387290,2.732337,Cedar
    ...

Labels are strings ("Cedar", "Pine") so both the C and Python classifiers can
handle them as opaque tokens. A fixed NumPy seed makes the file reproducible.
"""

import os
import sys
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(__file__), "trees.csv")


def generate_data(n_samples=200, seed=0):
    rng = np.random.default_rng(seed)

    half = n_samples // 2

    # Western Red Cedar: shorter, much thicker trunk.
    cedars_height = rng.normal(loc=22.0, scale=3.2, size=half)
    cedars_width  = rng.normal(loc=3.2,  scale=0.4, size=half)
    cedars = np.column_stack((cedars_height, cedars_width))

    # Lodgepole Pine: taller, slim trunk.
    pines_height = rng.normal(loc=28.0, scale=4.0, size=half)
    pines_width  = rng.normal(loc=0.4,  scale=0.1, size=half)
    pines = np.column_stack((pines_height, pines_width))

    X = np.vstack((cedars, pines))
    labels = ["Cedar"] * half + ["Pine"] * half

    # Shuffle rows so the file isn't sorted by class.
    idx = rng.permutation(len(labels))
    X = X[idx]
    labels = [labels[i] for i in idx]

    with open(DATA_PATH, "w") as f:
        f.write("height,width,label\n")
        for (h, w), lbl in zip(X, labels):
            f.write(f"{h:.6f},{w:.6f},{lbl}\n")

    print(f"Wrote {DATA_PATH}  ({n_samples} samples, seed={seed})")


if __name__ == "__main__":
    n    = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    generate_data(n_samples=n, seed=seed)
