import numpy as np
from data import generate_data, load_data, train_test_split, LABELS
from knn import KNN, find_best_k


def print_menu():
    print("\n========== KNN Tree Classifier ==========")
    print("  I.   Generate a new dataset")
    print("  II.  Explore the dataset")
    print("  III. Find best k and evaluate accuracy")
    print("  IV.  Classify a new tree")
    print("  V.   Quit")
    print("=========================================")


def explore(X, y):
    """Print basic stats about the loaded dataset."""
    print(f"\nDataset: {len(y)} samples, {X.shape[1]} features (height, width)")
    print(f"  Western Red Cedar (class 0): {int((y == 0).sum())} samples")
    print(f"  Lodgepole Pine    (class 1): {int((y == 1).sum())} samples")
    print(f"\n  Height — min {X[:,0].min():.2f}m  max {X[:,0].max():.2f}m  mean {X[:,0].mean():.2f}m")
    print(f"  Width  — min {X[:,1].min():.2f}m  max {X[:,1].max():.2f}m  mean {X[:,1].mean():.2f}m")


def evaluate(X, y):
    """Split, find best k, and report final test accuracy."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(f"\nTrain size: {len(y_train)}   Test size: {len(y_test)}")

    best_k, best_acc = find_best_k(X_train, y_train, X_test, y_test)
    return best_k


def classify_new_tree(X_train, y_train, k):
    """Ask the user for a tree's height and width and predict its species."""
    print("\n--- Classify a new tree ---")
    try:
        height = float(input("Enter tree height (m): "))
        width  = float(input("Enter trunk width  (m): "))
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return

    model = KNN(k=k)
    model.fit(X_train, y_train)

    new_tree = np.array([[height, width]])
    pred     = model.predict(new_tree)[0]
    species  = LABELS[int(pred)]

    print(f"\nPredicted species: {species}  (k={k})")


def main():
    k = 3   # default k until the user runs option III

    while True:
        print_menu()
        choice = input("Choose an option: ").strip().upper()

        if choice == "I":
            n = input("Number of samples (default 200): ").strip()
            n = int(n) if n.isdigit() else 200
            generate_data(n_samples=n)

        elif choice == "II":
            try:
                X, y = load_data()
                explore(X, y)
            except FileNotFoundError as e:
                print(f"\n{e}")

        elif choice == "III":
            try:
                X, y = load_data()
                k = evaluate(X, y)
            except FileNotFoundError as e:
                print(f"\n{e}")

        elif choice == "IV":
            try:
                X, y = load_data()
                X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2)
                classify_new_tree(X_train, y_train, k)
            except FileNotFoundError as e:
                print(f"\n{e}")

        elif choice == "V":
            print("Bye!")
            break

        else:
            print("Unknown option — please enter I, II, III, IV, or V.")


if __name__ == "__main__":
    main()
