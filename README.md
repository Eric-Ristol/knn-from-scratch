# KNN Tree Classifier

K-Nearest Neighbours from scratch. Classifies trees as Western Red Cedar or Lodgepole Pine based on height and trunk diameter.

## Quick start

```bash
python main.py
```

Menu options:
- I: Generate dataset
- II: Explore data
- III: Find best k
- IV: Classify a tree
- V: Quit

## How KNN works

For each new tree:
1. Calculate distance to every training tree (Euclidean distance)
2. Find k nearest trees
3. Vote on the species

No training phase -- just stores the data and does the work at prediction time.

## Files

| File | What it does |
|------|-------------|
| `data.py` | Generate tree measurements, save to CSV, train/test split |
| `knn.py` | KNN class (fit, predict, accuracy) in pure NumPy |
| `main.py` | CLI menu |

## Choosing k

Run option III to find the k that gives highest test accuracy. Too small = overfits to noise, too large = misses details.
