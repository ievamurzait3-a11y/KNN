import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


def load(path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
        return d[b"data"], np.array(d[b"labels"])

Xtr, ytr = [], []
for i in range(1, 6):
    X, y = load(f"C:/Users/ievas/OneDrive/KNN/KNN_data/data_batch_{i}")
    Xtr.append(X)
    ytr.append(y)
X = np.vstack(Xtr)[:5000]
y = np.hstack(ytr)[:5000]

X_test, y_test = load("C:/Users/ievas/OneDrive/KNN/KNN_data/test_batch")
X_test = X_test[:1000]
y_test = y_test[:1000]

kf = KFold(n_splits=5, shuffle=True, random_state=42)
k_values = [1, 3, 5, 7, 9]
metrics = ["manhattan", "euclidean"]
results = {"manhattan": [], "euclidean": []}

best_acc = 0.0
best_k = 0
best_metric = ""

for metric in metrics:
    for k in k_values:
        fold_scores = []

        for train_idx, val_idx in kf.split(X):
            model = KNeighborsClassifier(n_neighbors=k, metric=metric)
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[val_idx])
            acc = accuracy_score(y[val_idx], pred) * 100
            fold_scores.append(acc)

        mean_acc = np.mean(fold_scores)
        print(f"K={k}, Metric={metric}, Accuracy={mean_acc:.2f}%")
        results[metric].append(fold_scores)

        if mean_acc > best_acc:
            best_acc = mean_acc
            best_k = k
            best_metric = metric

plt.figure()
for metric, color in [("manhattan", "blue"), ("euclidean", "red")]:
    means = []
    stds = []

    for i, k in enumerate(k_values):
        fold_scores = results[metric][i]
        means.append(np.mean(fold_scores))
        stds.append(np.std(fold_scores))
        plt.scatter([k]*len(fold_scores), fold_scores,
                    color=color, alpha=0.5)

    plt.errorbar(
        k_values,
        means,
        yerr=stds,
        color=color,
        marker="o",
        capsize=5,
        linewidth=2,
        label=f"{metric} (mean ± std)"
    )

plt.title("5-Fold Cross Validation Accuracy (%)")
plt.xlabel("K value")
plt.ylabel("Accuracy (%)")
plt.ylim(22, 35)
plt.legend()
plt.grid()

print("\nBest result: ")
print(f"K={best_k}, Metric={best_metric}, Accuracy={best_acc:.2f}%")
final_model = KNeighborsClassifier(n_neighbors=best_k, metric=best_metric)
final_model.fit(X, y)

y_pred = final_model.predict(X_test)
print("\nFinal accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
plt.show()
