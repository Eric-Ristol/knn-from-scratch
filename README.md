# KNN — from scratch in C and Python

Two implementations of the same K-Nearest Neighbors classifier, in C and in Python, sharing the same dataset, the same CLI, and the same deterministic shuffle. Given the same seed they produce bit-for-bit identical output.

## Repository layout

| File | Purpose |
|------|---------|
| `knn.c` | C implementation — full classifier in a single file. |
| `knn.py` | Python implementation — pure stdlib, mirrors `knn.c` exactly. |
| `data.py` | Utility to regenerate `trees.csv`. |
| `trees.csv` | Synthetic dataset: 200 samples, 2 features (height, width), 2 classes (Cedar, Pine). |
| `Makefile` | Builds the C binary and provides a `make run` shortcut. |

## The dataset

`trees.csv` is a synthetic two-feature, two-class dataset — Western Red Cedar (short, thick trunk) vs Lodgepole Pine (tall, slim trunk). It is checked in and reproducible: running `python data.py` with the default seed produces the same file.

```
height,width,label
24.049352,3.134282,Cedar
26.617310,0.423567,Pine
23.273268,0.365969,Pine
...
```

To regenerate with different parameters:

```sh
python data.py 500 7      # 500 samples, seed 7
```

## Build and run

### C

```sh
make                      # builds ./knn
make run                  # runs ./knn trees.csv 3 0.2 42
./knn trees.csv 3 0.2 42  # run manually
```

Or without `make`:

```sh
gcc -O2 -Wall -Wextra -std=c11 -o knn knn.c -lm
```

### Python

```sh
python knn.py trees.csv 3 0.2 42
```

No third-party dependencies — standard library only.

## CLI

Both programs share the exact same argument list:

| Pos | Name        | Default      | Meaning                               |
|-----|-------------|--------------|---------------------------------------|
| 1   | csv         | *(required)* | Path to the CSV dataset.              |
| 2   | k           | `3`          | Number of neighbors.                  |
| 3   | test_ratio  | `0.2`        | Fraction of samples used for testing. |
| 4   | seed        | time-derived | RNG seed for the train/test shuffle.  |

## How the algorithm works

1. **Load CSV.** Comma-separated, optional header row. All but the last column are numeric features; the last column is a string label.
2. **Shuffle** with Fisher-Yates, driven by a fixed xorshift32 PRNG seeded from the `seed` argument.
3. **Split** into train / test using `test_ratio` (rounded to the nearest integer, minimum 1 test sample).
4. **Predict** each test sample by computing Euclidean distance to every training sample, keeping the `K` closest via a partial insertion sort (stable on ties), and taking a majority vote of their labels. On vote ties, the label of the nearest neighbor wins.
5. **Report** per-sample predictions and overall accuracy.

## Parity between the two implementations

The two programs are meant to be interchangeable. To enforce that, both sides implement the same building blocks with matching behavior:

| Step               | C (`knn.c`)              | Python (`knn.py`)         |
|--------------------|--------------------------|---------------------------|
| PRNG               | `prng_next()` — xorshift32 | `XorShift32.next_u32()` — same algorithm |
| Shuffle            | Fisher-Yates, `j = r % (i+1)` | Fisher-Yates, `j = r % (i+1)` |
| Split size         | `int(n * test_ratio + 0.5)` | `int(n * test_ratio + 0.5)` |
| Distance           | Euclidean                  | Euclidean                   |
| Top-K selection    | Partial insertion sort, stable on ties | Same |
| Tie-break in vote  | First (nearest) label wins | First (nearest) label wins  |

Same `csv` + same `seed` → same shuffle → same train/test split → same predictions → same accuracy. You can verify directly:

```sh
make
./knn trees.csv 3 0.2 42 > c.out
python knn.py trees.csv 3 0.2 42 > py.out
diff c.out py.out          # empty output = identical
```

### Why not use C's `rand()` or NumPy's RNG?

Both would break parity. `rand()` is implementation-defined (different on macOS vs glibc), and NumPy's RNGs use algorithms that don't exist in the C stdlib. Implementing the same tiny xorshift32 on both sides is the cheapest way to make "same seed → same output" a hard guarantee rather than a vague claim.

## Example output

```
dataset : trees.csv
samples : 200  (train=160, test=40)
features: 2
k       : 3
seed    : 42

idx  true              predicted         match
---  ----------------  ----------------  -----
  0  Cedar             Cedar             yes
  1  Pine              Pine              yes
  ...
 39  Cedar             Cedar             yes

accuracy: 40/40 = 1.0000
```

The trees dataset is well-separated by design, so most reasonable `(k, seed)` combinations score in the high 90s. Try `./knn trees.csv 7 0.2 123` to see a case where one test sample ends up on the wrong side of the boundary.

## Limits (C only)

Compile-time caps in `knn.c`:

```
MAX_LINE      1024
MAX_FEATURES    64
MAX_SAMPLES   5000
MAX_CLASSES    128
LABEL_LEN       64
```

Python has no equivalent caps.
