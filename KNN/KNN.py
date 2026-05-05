import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def load(path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
        return d[b"data"], np.array(d[b"labels"])



Xtr, ytr = [], []

for i in range(1, 6):
    X, y = load(f"C:/Users/ievas/OneDrive/KNN/KNN_data/data_batch_{i}")
    Xtr.append(X)
    ytr.append(y)

X_train_full = np.vstack(Xtr)
y_train_full = np.hstack(ytr)


X_train = X_train_full[:5000]
y_train = y_train_full[:5000]


X_test, y_test = load(f"C:/Users/ievas/OneDrive/KNN/KNN_data/test_batch")
X_test = X_test[:1000]
y_test = y_test[:1000]


k_values = [1, 3, 5, 7, 9]
metrics = ["manhattan", "euclidean"]

results = []

best_acc = 0
best_model = None
best_k = None
best_metric = None

for metric in metrics:
    for k in k_values:

        model = KNeighborsClassifier(
            n_neighbors=k,
            metric=metric,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        results.append((k, metric, acc))

        print(f"K={k}, Metric={metric}, Accuracy={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_k = k
            best_metric = metric



print("\nBEST MODEL")
print(f"K={best_k}, Metric={best_metric}, Accuracy={best_acc:.4f}")

y_best = best_model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_best))