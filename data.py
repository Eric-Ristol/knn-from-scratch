import numpy as np
import os

"""
Generates the data of a Pacific Northwest forest.

Western Red Cedar   Height (m): 22.0    Width (m): 3.2
Lodgepole Pine      Height (m): 28.0    Width (m): 0.4
"""

DATA_PATH = os.path.join(os.path.dirname(__file__), "trees.csv")

LABELS = {0: "Western Red Cedar", 1: "Lodgepole Pine"}


def generate_data(n_samples=100):
    """
    Generates synthetic tree measurements and saves them to trees.csv.

    Class 0 = "Western Red Cedar"
    Class 1 = "Lodgepole Pine"

    loc = mean, scale = standard deviation
    """

    # --- Western Red Cedar ---
    cedars_height = np.random.normal(loc=22.0, scale=3.2, size=n_samples // 2)
    cedars_width  = np.random.normal(loc=3.2,  scale=0.4, size=n_samples // 2)

    # Combine height and width into feature pairs
    cedars   = np.column_stack((cedars_height, cedars_width))
    l_cedars = np.zeros(n_samples // 2)

    # --- Lodgepole Pine ---
    pine_height = np.random.normal(loc=28.0, scale=4.0, size=n_samples // 2)
    pine_width  = np.random.normal(loc=0.4,  scale=0.1, size=n_samples // 2)
    pines   = np.column_stack((pine_height, pine_width))
    l_pines = np.ones(n_samples // 2)

    # Stack both classes together and shuffle
    X = np.vstack((cedars, pines))
    y = np.concatenate((l_cedars, l_pines))

    idx = np.random.permutation(len(y))
    X, y = X[idx], y[idx]

    # Save to CSV (header: height,width,label)
    data = np.column_stack((X, y))
    np.savetxt(DATA_PATH, data, delimiter=",", header="height,width,label", comments="")
    print(f"Dataset saved to {DATA_PATH}  ({n_samples} samples)")

    return X, y


def load_data():
    """Loads the CSV created by generate_data and returns X, y arrays."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"No dataset found at {DATA_PATH}. "
            "Run option I first to generate it."
        )
    data = np.loadtxt(DATA_PATH, delimiter=",", skiprows=1)
    X = data[:, :2]   # height, width
    y = data[:, 2]    # label (0 or 1)
    return X, y


def train_test_split(X, y, test_size=0.2):
    """
    Splits X and y into train/test sets.
    Uses a fixed random seed so results are reproducible across runs.
    """
    n = len(y)
    n_test = int(n * test_size)

    rng = np.random.default_rng(seed=42)
    idx = rng.permutation(n)

    test_idx  = idx[:n_test]
    train_idx = idx[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
