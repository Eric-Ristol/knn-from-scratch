# KNN Tree Classifier

A from-scratch implementation of the K-Nearest Neighbours algorithm, applied to classify trees in a Pacific Northwest forest.

Two species:
- **Western Red Cedar** — shorter (~22 m), wider trunk (~3.2 m)
- **Lodgepole Pine** — taller (~28 m), thinner trunk (~0.4 m)

---

## What's implemented

| File        | What it does |
|-------------|-------------|
| `data.py`   | Generates synthetic tree measurements, saves to `trees.csv`, provides train/test split |
| `knn.py`    | `KNN` class (fit, predict, accuracy) + `find_best_k` helper — **no sklearn**, pure NumPy |
| `main.py`   | Interactive CLI menu |

## How to run

```bash
python main.py
```

Menu options:

```
I.   Generate a new dataset
II.  Explore the dataset (counts + feature stats)
III. Find best k and evaluate accuracy
IV.  Classify a new tree
V.   Quit
```

A typical run is: **I → II → III → IV**.

## How KNN works (from scratch)

For each new point to classify:

1. Compute **Euclidean distance** to every point in the training set.
2. Pick the **k nearest** neighbours.
3. **Majority vote** among their labels → predicted class.

No training phase — KNN is a *lazy learner* that stores the training set and does all the work at prediction time.

## Choosing k

Run option **III** to sweep k from 1 to 15 and see which value gives the highest test accuracy. A small k is sensitive to noise; a large k smooths the decision boundary but may underfit.

## What I learned

- How to implement Euclidean distance and nearest-neighbour search without sklearn.
- Why KNN is a *lazy* learner and what that means for memory vs speed.
- How the choice of k trades off bias vs variance.
- Why features on very different scales (height vs width here) can skew distances — and why feature scaling matters.

## Author

Eric Ristol — 1st year Bachelor in Artificial Intelligence, UAB / La Salle.
