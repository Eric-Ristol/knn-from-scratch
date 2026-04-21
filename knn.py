#!/usr/bin/env python3
"""
knn.py -- K-Nearest Neighbors classifier in pure Python.

Mirror of knn.c in this repository. Both programs implement the identical
algorithm, expose the identical CLI, and use the identical deterministic
shuffle (xorshift32), so running them with the same seed on the same CSV
produces bit-for-bit identical output.

Usage:
    python knn.py <csv> [k=3] [test_ratio=0.2] [seed]

        arg1 = CSV file     (required)
        arg2 = K            (default 3)
        arg3 = test ratio   (default 0.2)
        arg4 = random seed  (default: derived from time.time())

CSV format: comma-separated, optional header row, numeric features followed by
a string label in the last column. Example (trees.csv):

    height,width,label
    24.049352,3.134282,Cedar
    26.617310,0.423567,Pine
    ...
"""

import math
import sys
import time


# ---------- xorshift32 PRNG ------------------------------------------------
# Portable replacement for C's rand()/srand(). Matches the xorshift32 in
# knn.c bit-for-bit, so the shuffle below produces the same sequence in
# both languages given the same seed.

_MASK32 = 0xFFFFFFFF


class XorShift32:
    def __init__(self, seed):
        seed &= _MASK32
        self.state = seed if seed != 0 else 1  # xorshift requires non-zero

    def next_u32(self):
        x = self.state
        x = (x ^ ((x << 13) & _MASK32)) & _MASK32
        x = (x ^ (x >> 17)) & _MASK32
        x = (x ^ ((x << 5)  & _MASK32)) & _MASK32
        self.state = x
        return x


# ---------- CSV loader -----------------------------------------------------

def _looks_like_header(line):
    s = line.lstrip(" \t")
    if s[:1] in ("+", "-"):
        s = s[1:]
    return not (s[:1].isdigit() or s[:1] == ".")


def load_csv(path):
    """Return (samples, n_features) where samples is a list of (features, label)."""
    samples = []
    n_features = None
    with open(path, "r") as f:
        first_line = True
        for line in f:
            line = line.strip()
            if not line:
                continue
            if first_line and _looks_like_header(line):
                first_line = False
                continue
            first_line = False

            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue

            nf = len(parts) - 1
            if n_features is None:
                n_features = nf
            elif nf != n_features:
                print(f"warning: row has {nf} features (expected {n_features}); skipping",
                      file=sys.stderr)
                continue

            feats = [float(x) for x in parts[:nf]]
            label = parts[nf]
            samples.append((feats, label))

    return samples, (n_features or 0)


# ---------- shuffle (Fisher-Yates, driven by xorshift32) -------------------

def shuffle(arr, rng):
    for i in range(len(arr) - 1, 0, -1):
        r = rng.next_u32()
        j = r % (i + 1)
        arr[i], arr[j] = arr[j], arr[i]


# ---------- distance -------------------------------------------------------

def euclidean(a, b):
    s = 0.0
    for ai, bi in zip(a, b):
        d = ai - bi
        s += d * d
    return math.sqrt(s)


# ---------- K smallest via partial insertion sort --------------------------
# Stable on ties: a new sample with the same distance is inserted AFTER
# existing ones. knn.c's insert_neighbor uses the same rule.

def _insert_neighbor(neighbors, k, dist, idx):
    count = len(neighbors)
    if count < k:
        i = count
        neighbors.append(None)
        while i > 0 and neighbors[i - 1][0] > dist:
            neighbors[i] = neighbors[i - 1]
            i -= 1
        neighbors[i] = (dist, idx)
    elif dist < neighbors[k - 1][0]:
        i = k - 1
        while i > 0 and neighbors[i - 1][0] > dist:
            neighbors[i] = neighbors[i - 1]
            i -= 1
        neighbors[i] = (dist, idx)


# ---------- majority vote --------------------------------------------------
# On ties the first label encountered (nearest in distance) wins. knn.c does
# the same.

def _majority_vote(neighbors, train):
    counts = {}
    order = []
    for _d, idx in neighbors:
        lbl = train[idx][1]
        if lbl not in counts:
            counts[lbl] = 0
            order.append(lbl)
        counts[lbl] += 1

    best = order[0]
    for lbl in order[1:]:
        if counts[lbl] > counts[best]:
            best = lbl
    return best


# ---------- classify one sample -------------------------------------------

def knn_predict(train, query_feats, k):
    neighbors = []  # list of (distance, index), sorted ascending by distance
    for idx, (feats, _lbl) in enumerate(train):
        d = euclidean(feats, query_feats)
        _insert_neighbor(neighbors, k, d, idx)
    return _majority_vote(neighbors, train)


# ---------- main -----------------------------------------------------------

def main(argv):
    if len(argv) < 2:
        print(f"usage: {argv[0]} <csv> [k=3] [test_ratio=0.2] [seed]",
              file=sys.stderr)
        return 1

    csv_path   = argv[1]
    k          = int(argv[2])   if len(argv) > 2 else 3
    test_ratio = float(argv[3]) if len(argv) > 3 else 0.2
    seed       = int(argv[4])   if len(argv) > 4 else int(time.time())
    seed &= _MASK32

    if k <= 0:
        print("error: k must be > 0", file=sys.stderr)
        return 1
    if not (0.0 < test_ratio < 1.0):
        print("error: test_ratio must be in (0, 1)", file=sys.stderr)
        return 1

    try:
        data, n_features = load_csv(csv_path)
    except FileNotFoundError:
        print(f"error: cannot open '{csv_path}'", file=sys.stderr)
        return 1

    n = len(data)
    if n == 0:
        print("error: no samples loaded", file=sys.stderr)
        return 1

    rng = XorShift32(seed)
    shuffle(data, rng)

    n_test  = int(n * test_ratio + 0.5)
    if n_test < 1:
        n_test = 1
    n_train = n - n_test
    if n_train < k:
        print(f"error: not enough training samples ({n_train}) for k={k}",
              file=sys.stderr)
        return 1

    train = data[:n_train]
    test  = data[n_train:]

    print(f"dataset : {csv_path}")
    print(f"samples : {n}  (train={n_train}, test={n_test})")
    print(f"features: {n_features}")
    print(f"k       : {k}")
    print(f"seed    : {seed}\n")

    correct = 0
    print("idx  true              predicted         match")
    print("---  ----------------  ----------------  -----")
    for i, (feats, true_lbl) in enumerate(test):
        pred = knn_predict(train, feats, k)
        ok = (pred == true_lbl)
        if ok:
            correct += 1
        print(f"{i:3d}  {true_lbl:<16s}  {pred:<16s}  {'yes' if ok else 'NO'}")

    acc = correct / n_test
    print(f"\naccuracy: {correct}/{n_test} = {acc:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
