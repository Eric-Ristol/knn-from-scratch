import numpy as np
from collections import Counter


class KNN:
    """
    K-Nearest Neighbours classifier implemented from scratch.

    Steps for prediction:
      1. Compute the Euclidean distance from the query point to every training point.
      2. Pick the k closest neighbours.
      3. Return the majority class among those neighbours.
    """

    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Store the training set — KNN is a lazy learner (no actual training)."""
        self.X_train = X_train
        self.y_train = y_train

    def _euclidean_distance(self, a, b):
        """Euclidean distance between two points."""
        return np.sqrt(np.sum((a - b) ** 2))

    def _predict_one(self, x):
        """Classify a single sample x."""
        # Compute distances to all training points
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]

        # Get the indices of the k nearest neighbours (ascending distance)
        k_nearest_indices = np.argsort(distances)[: self.k]

        # Retrieve their labels
        k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]

        # Majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def predict(self, X):
        """Classify every sample in X and return an array of predictions."""
        return np.array([self._predict_one(x) for x in X])

    def accuracy(self, X, y):
        """Return classification accuracy on a labelled set."""
        preds = self.predict(X)
        return np.mean(preds == y)


def find_best_k(X_train, y_train, X_test, y_test, k_range=range(1, 16)):
    """
    Tries every k in k_range and returns the value that maximises test accuracy.
    Also prints a small comparison table.
    """
    print(f"\n{'k':>4}  {'Accuracy':>10}")
    print("-" * 18)

    best_k   = None
    best_acc = -1.0

    for k in k_range:
        model = KNN(k=k)
        model.fit(X_train, y_train)
        acc = model.accuracy(X_test, y_test)
        print(f"{k:>4}  {acc * 100:>9.2f}%")

        if acc > best_acc:
            best_acc = acc
            best_k   = k

    print(f"\nBest k = {best_k}  (accuracy {best_acc * 100:.2f}%)")
    return best_k, best_acc
