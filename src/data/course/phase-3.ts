import type { CourseExercise } from "../types";

export const phase3Exercises: CourseExercise[] = [
  {
    id: "svm-intro",
    type: "course",
    phase: 3,
    order: 1,
    title: "Support Vector Machines",
    description:
      "Learn SVM - a powerful algorithm that finds the optimal boundary between classes.",
    difficulty: "hard",
    concepts: ["SVC", "support vectors", "margin", "C parameter"],
    hints: [
      "SVM finds the hyperplane that maximizes margin between classes",
      "C controls the trade-off between margin and misclassification",
      "Larger C = less margin, fewer misclassifications",
    ],
    docsUrl: ["https://scikit-learn.org/stable/modules/svm.html#classification"],
    code: `from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

# Load data (use only 2 classes for visualization)
iris = load_iris()
X = iris.data[iris.target != 2]
y = iris.target[iris.target != 2]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TODO: Create SVM with linear kernel and C=1.0
svm_linear = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', None)  # <- SVC(kernel='linear', C=1.0)
])

# TODO: Fit and score
None  # <- svm_linear.fit(X_train, y_train)
linear_score = None  # <- svm_linear.score(X_test, y_test)

print(f"Linear SVM accuracy: {linear_score:.3f}")
print()

# Compare different C values
print("Effect of C parameter (linear kernel):")
for C in [0.01, 0.1, 1.0, 10, 100]:
    svm = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='linear', C=C))
    ])
    svm.fit(X_train, y_train)
    acc = svm.score(X_test, y_test)

    # Get number of support vectors
    n_sv = svm.named_steps['svc'].n_support_.sum()
    print(f"  C={C:5}: accuracy={acc:.3f}, support vectors={n_sv}")

print()
print("Note: More support vectors = softer margin")

result = f"SVM accuracy: {linear_score:.3f}"
result`,
    solution: `from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

# Load data (use only 2 classes for visualization)
iris = load_iris()
X = iris.data[iris.target != 2]
y = iris.target[iris.target != 2]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create SVM with linear kernel and C=1.0
svm_linear = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='linear', C=1.0))
])

# Fit and score
svm_linear.fit(X_train, y_train)
linear_score = svm_linear.score(X_test, y_test)

print(f"Linear SVM accuracy: {linear_score:.3f}")
print()

# Compare different C values
print("Effect of C parameter (linear kernel):")
for C in [0.01, 0.1, 1.0, 10, 100]:
    svm = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='linear', C=C))
    ])
    svm.fit(X_train, y_train)
    acc = svm.score(X_test, y_test)

    # Get number of support vectors
    n_sv = svm.named_steps['svc'].n_support_.sum()
    print(f"  C={C:5}: accuracy={acc:.3f}, support vectors={n_sv}")

print()
print("Note: More support vectors = softer margin")

result = f"SVM accuracy: {linear_score:.3f}"
result`,
  },
  {
    id: "kernel-tricks",
    type: "course",
    phase: 3,
    order: 2,
    title: "SVM Kernel Tricks",
    description:
      "Learn how kernels allow SVMs to handle non-linear decision boundaries.",
    difficulty: "hard",
    concepts: ["kernel", "RBF", "polynomial", "gamma", "non-linear"],
    hints: [
      "RBF kernel: similarity decreases with distance (controlled by gamma)",
      "Polynomial kernel: decision boundary is a polynomial of specified degree",
      "Higher gamma = more complex boundary (may overfit)",
    ],
    docsUrl: ["https://scikit-learn.org/stable/modules/svm.html#kernel-functions"],
    code: `from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

# Create non-linearly separable data
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Testing different kernels on 'moons' dataset:")
print("-" * 45)

# TODO: Test different kernels
kernels = ['linear', 'poly', 'rbf']

for kernel in kernels:
    # TODO: Create SVM with current kernel
    svm = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', None)  # <- SVC(kernel=kernel, random_state=42)
    ])

    svm.fit(X_train, y_train)
    train_score = svm.score(X_train, y_train)
    test_score = svm.score(X_test, y_test)

    print(f"{kernel:8}: train={train_score:.3f}, test={test_score:.3f}")

print()

# Explore gamma parameter for RBF kernel
print("Effect of gamma (RBF kernel):")
for gamma in [0.1, 1, 10, 100]:
    svm = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='rbf', gamma=gamma, random_state=42))
    ])
    svm.fit(X_train, y_train)
    train_score = svm.score(X_train, y_train)
    test_score = svm.score(X_test, y_test)
    print(f"  gamma={gamma:4}: train={train_score:.3f}, test={test_score:.3f}")

print()
print("Note: Very high gamma can overfit (perfect train, poor test)")

result = "Kernel comparison complete"
result`,
    solution: `from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

# Create non-linearly separable data
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Testing different kernels on 'moons' dataset:")
print("-" * 45)

# Test different kernels
kernels = ['linear', 'poly', 'rbf']

for kernel in kernels:
    # Create SVM with current kernel
    svm = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel=kernel, random_state=42))
    ])

    svm.fit(X_train, y_train)
    train_score = svm.score(X_train, y_train)
    test_score = svm.score(X_test, y_test)

    print(f"{kernel:8}: train={train_score:.3f}, test={test_score:.3f}")

print()

# Explore gamma parameter for RBF kernel
print("Effect of gamma (RBF kernel):")
for gamma in [0.1, 1, 10, 100]:
    svm = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='rbf', gamma=gamma, random_state=42))
    ])
    svm.fit(X_train, y_train)
    train_score = svm.score(X_train, y_train)
    test_score = svm.score(X_test, y_test)
    print(f"  gamma={gamma:4}: train={train_score:.3f}, test={test_score:.3f}")

print()
print("Note: Very high gamma can overfit (perfect train, poor test)")

result = "Kernel comparison complete"
result`,
  },
  {
    id: "gradient-boosting",
    type: "course",
    phase: 3,
    order: 3,
    title: "Gradient Boosting",
    description:
      "Learn Gradient Boosting - a powerful ensemble that builds trees sequentially to correct errors.",
    difficulty: "hard",
    concepts: ["GradientBoostingClassifier", "boosting", "learning_rate", "n_estimators"],
    hints: [
      "Boosting builds trees sequentially, each correcting previous errors",
      "learning_rate controls how much each tree contributes",
      "Lower learning_rate usually needs more estimators",
    ],
    docsUrl: ["https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting"],
    code: `from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import numpy as np

# Load data
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42
)

# TODO: Create Gradient Boosting classifier
gb = None  # <- GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

# TODO: Fit and score
None  # <- gb.fit(X_train, y_train)
gb_score = None  # <- gb.score(X_test, y_test)

print(f"Gradient Boosting accuracy: {gb_score:.3f}")
print()

# Compare with Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_score = rf.score(X_test, y_test)

print("Comparison:")
print(f"  Gradient Boosting: {gb_score:.3f}")
print(f"  Random Forest:     {rf_score:.3f}")
print()

# Effect of learning_rate
print("Effect of learning_rate (100 estimators):")
for lr in [0.01, 0.1, 0.5, 1.0]:
    gb_temp = GradientBoostingClassifier(
        n_estimators=100, learning_rate=lr, random_state=42
    )
    gb_temp.fit(X_train, y_train)
    train_score = gb_temp.score(X_train, y_train)
    test_score = gb_temp.score(X_test, y_test)
    print(f"  lr={lr}: train={train_score:.3f}, test={test_score:.3f}")

result = f"Gradient Boosting: {gb_score:.3f}"
result`,
    solution: `from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import numpy as np

# Load data
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42
)

# Create Gradient Boosting classifier
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

# Fit and score
gb.fit(X_train, y_train)
gb_score = gb.score(X_test, y_test)

print(f"Gradient Boosting accuracy: {gb_score:.3f}")
print()

# Compare with Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_score = rf.score(X_test, y_test)

print("Comparison:")
print(f"  Gradient Boosting: {gb_score:.3f}")
print(f"  Random Forest:     {rf_score:.3f}")
print()

# Effect of learning_rate
print("Effect of learning_rate (100 estimators):")
for lr in [0.01, 0.1, 0.5, 1.0]:
    gb_temp = GradientBoostingClassifier(
        n_estimators=100, learning_rate=lr, random_state=42
    )
    gb_temp.fit(X_train, y_train)
    train_score = gb_temp.score(X_train, y_train)
    test_score = gb_temp.score(X_test, y_test)
    print(f"  lr={lr}: train={train_score:.3f}, test={test_score:.3f}")

result = f"Gradient Boosting: {gb_score:.3f}"
result`,
  },
  {
    id: "voting-classifier",
    type: "course",
    phase: 3,
    order: 4,
    title: "Voting Classifiers",
    description:
      "Learn to combine multiple different models using voting for better predictions.",
    difficulty: "hard",
    concepts: ["VotingClassifier", "hard voting", "soft voting", "ensemble diversity"],
    hints: [
      "Hard voting: majority vote of predictions",
      "Soft voting: average of predicted probabilities",
      "Diverse models (different algorithms) often work best together",
    ],
    docsUrl: ["https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier"],
    code: `from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Load data
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42
)

# Define individual classifiers
clf1 = LogisticRegression(max_iter=5000, random_state=42)
clf2 = RandomForestClassifier(n_estimators=50, random_state=42)
clf3 = SVC(probability=True, random_state=42)
clf4 = KNeighborsClassifier(n_neighbors=5)

# TODO: Create VotingClassifier with soft voting
voting_clf = VotingClassifier(
    estimators=[
        ('lr', clf1),
        ('rf', clf2),
        ('svc', clf3),
        ('knn', clf4)
    ],
    voting=None  # <- 'soft'
)

# Compare individual vs ensemble
print("Individual classifier scores (CV):")
for clf, name in [(clf1, 'Logistic'), (clf2, 'RandomForest'),
                   (clf3, 'SVC'), (clf4, 'KNN')]:
    scores = cross_val_score(clf, cancer.data, cancer.target, cv=5)
    print(f"  {name:12}: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")

# TODO: Fit voting classifier and get CV scores
voting_scores = None  # <- cross_val_score(voting_clf, cancer.data, cancer.target, cv=5)

print(f"\\nVoting Ensemble: {voting_scores.mean():.3f} (+/- {voting_scores.std()*2:.3f})")
print()

# Fit and evaluate on test set
voting_clf.fit(X_train, y_train)
print(f"Test set accuracy: {voting_clf.score(X_test, y_test):.3f}")

# Compare hard vs soft voting
voting_hard = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('svc', clf3), ('knn', clf4)],
    voting='hard'
)
voting_hard.fit(X_train, y_train)
print(f"Hard voting test accuracy: {voting_hard.score(X_test, y_test):.3f}")

result = f"Voting ensemble CV: {voting_scores.mean():.3f}"
result`,
    solution: `from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Load data
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42
)

# Define individual classifiers
clf1 = LogisticRegression(max_iter=5000, random_state=42)
clf2 = RandomForestClassifier(n_estimators=50, random_state=42)
clf3 = SVC(probability=True, random_state=42)
clf4 = KNeighborsClassifier(n_neighbors=5)

# Create VotingClassifier with soft voting
voting_clf = VotingClassifier(
    estimators=[
        ('lr', clf1),
        ('rf', clf2),
        ('svc', clf3),
        ('knn', clf4)
    ],
    voting='soft'
)

# Compare individual vs ensemble
print("Individual classifier scores (CV):")
for clf, name in [(clf1, 'Logistic'), (clf2, 'RandomForest'),
                   (clf3, 'SVC'), (clf4, 'KNN')]:
    scores = cross_val_score(clf, cancer.data, cancer.target, cv=5)
    print(f"  {name:12}: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")

# Fit voting classifier and get CV scores
voting_scores = cross_val_score(voting_clf, cancer.data, cancer.target, cv=5)

print(f"\\nVoting Ensemble: {voting_scores.mean():.3f} (+/- {voting_scores.std()*2:.3f})")
print()

# Fit and evaluate on test set
voting_clf.fit(X_train, y_train)
print(f"Test set accuracy: {voting_clf.score(X_test, y_test):.3f}")

# Compare hard vs soft voting
voting_hard = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('svc', clf3), ('knn', clf4)],
    voting='hard'
)
voting_hard.fit(X_train, y_train)
print(f"Hard voting test accuracy: {voting_hard.score(X_test, y_test):.3f}")

result = f"Voting ensemble CV: {voting_scores.mean():.3f}"
result`,
  },
  {
    id: "pca",
    type: "course",
    phase: 3,
    order: 5,
    title: "PCA Dimensionality Reduction",
    description:
      "Learn Principal Component Analysis for reducing the number of features while preserving variance.",
    difficulty: "hard",
    concepts: ["PCA", "explained variance", "dimensionality reduction", "components"],
    hints: [
      "PCA finds directions of maximum variance",
      "explained_variance_ratio_ tells you how much info each component captures",
      "Use it for visualization (2D), speedup, or denoising",
    ],
    docsUrl: ["https://scikit-learn.org/stable/modules/decomposition.html#pca"],
    code: `from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt

# Load digits dataset (64 features)
digits = load_digits()
X, y = digits.data, digits.target

print(f"Original shape: {X.shape}")
print()

# TODO: Create PCA with n_components=2 for visualization
pca_2d = None  # <- PCA(n_components=2)
X_2d = None  # <- pca_2d.fit_transform(X)

print(f"Reduced shape: {X_2d.shape}")
print(f"Explained variance (2 components): {pca_2d.explained_variance_ratio_.sum():.2%}")
print()

# Visualize in 2D
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', alpha=0.6)
plt.colorbar(scatter, label='Digit')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Digits Dataset - PCA (2 components)')
plt.show()

# Find optimal number of components
pca_full = PCA()
pca_full.fit(X)

cumsum = np.cumsum(pca_full.explained_variance_ratio_)
n_95 = np.argmax(cumsum >= 0.95) + 1
print(f"Components needed for 95% variance: {n_95}")
print()

# Compare classification with different n_components
print("Classification accuracy with different components:")
for n_comp in [10, 20, 30, 64]:
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_comp)),
        ('clf', LogisticRegression(max_iter=5000, random_state=42))
    ])
    scores = cross_val_score(pipe, X, y, cv=3)
    print(f"  {n_comp:2} components: {scores.mean():.3f}")

result = f"95% variance with {n_95} components"
result`,
    solution: `from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt

# Load digits dataset (64 features)
digits = load_digits()
X, y = digits.data, digits.target

print(f"Original shape: {X.shape}")
print()

# Create PCA with n_components=2 for visualization
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X)

print(f"Reduced shape: {X_2d.shape}")
print(f"Explained variance (2 components): {pca_2d.explained_variance_ratio_.sum():.2%}")
print()

# Visualize in 2D
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', alpha=0.6)
plt.colorbar(scatter, label='Digit')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Digits Dataset - PCA (2 components)')
plt.show()

# Find optimal number of components
pca_full = PCA()
pca_full.fit(X)

cumsum = np.cumsum(pca_full.explained_variance_ratio_)
n_95 = np.argmax(cumsum >= 0.95) + 1
print(f"Components needed for 95% variance: {n_95}")
print()

# Compare classification with different n_components
print("Classification accuracy with different components:")
for n_comp in [10, 20, 30, 64]:
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_comp)),
        ('clf', LogisticRegression(max_iter=5000, random_state=42))
    ])
    scores = cross_val_score(pipe, X, y, cv=3)
    print(f"  {n_comp:2} components: {scores.mean():.3f}")

result = f"95% variance with {n_95} components"
result`,
  },
  {
    id: "kmeans",
    type: "course",
    phase: 3,
    order: 6,
    title: "K-Means Clustering",
    description:
      "Learn K-Means clustering - an unsupervised algorithm for grouping similar data points.",
    difficulty: "hard",
    concepts: ["KMeans", "clustering", "centroids", "inertia", "elbow method"],
    hints: [
      "K-Means partitions data into k clusters based on distance to centroids",
      "inertia_ measures how tight clusters are (lower = better)",
      "Use elbow method to find optimal number of clusters",
    ],
    docsUrl: ["https://scikit-learn.org/stable/modules/clustering.html#k-means"],
    code: `from sklearn.datasets import make_blobs, load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import numpy as np
import matplotlib.pyplot as plt

# Create clusterable data
X, y_true = make_blobs(n_samples=300, centers=4, random_state=42)

print("K-Means Clustering Demo")
print("-" * 40)

# TODO: Create KMeans with k=4
kmeans = None  # <- KMeans(n_clusters=4, random_state=42, n_init=10)

# TODO: Fit and predict cluster labels
labels = None  # <- kmeans.fit_predict(X)

print(f"Cluster centers shape: {kmeans.cluster_centers_.shape}")
print(f"Inertia (within-cluster sum of squares): {kmeans.inertia_:.2f}")
print(f"Silhouette score: {silhouette_score(X, labels):.3f}")
print()

# Visualize
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='tab10')
plt.title('True Labels')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='red', marker='X', s=200, edgecolors='black')
plt.title('K-Means Clusters')

plt.tight_layout()
plt.show()

# Elbow method to find optimal k
print("Elbow Method Analysis:")
inertias = []
silhouettes = []
k_range = range(2, 10)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X, km.labels_))
    print(f"  k={k}: inertia={km.inertia_:.1f}, silhouette={silhouettes[-1]:.3f}")

result = f"K-Means with 4 clusters, silhouette={silhouette_score(X, labels):.3f}"
result`,
    solution: `from sklearn.datasets import make_blobs, load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import numpy as np
import matplotlib.pyplot as plt

# Create clusterable data
X, y_true = make_blobs(n_samples=300, centers=4, random_state=42)

print("K-Means Clustering Demo")
print("-" * 40)

# Create KMeans with k=4
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)

# Fit and predict cluster labels
labels = kmeans.fit_predict(X)

print(f"Cluster centers shape: {kmeans.cluster_centers_.shape}")
print(f"Inertia (within-cluster sum of squares): {kmeans.inertia_:.2f}")
print(f"Silhouette score: {silhouette_score(X, labels):.3f}")
print()

# Visualize
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='tab10')
plt.title('True Labels')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='red', marker='X', s=200, edgecolors='black')
plt.title('K-Means Clusters')

plt.tight_layout()
plt.show()

# Elbow method to find optimal k
print("Elbow Method Analysis:")
inertias = []
silhouettes = []
k_range = range(2, 10)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X, km.labels_))
    print(f"  k={k}: inertia={km.inertia_:.1f}, silhouette={silhouettes[-1]:.3f}")

result = f"K-Means with 4 clusters, silhouette={silhouette_score(X, labels):.3f}"
result`,
  },
  {
    id: "dbscan",
    type: "course",
    phase: 3,
    order: 7,
    title: "DBSCAN Clustering",
    description:
      "Learn DBSCAN - a density-based clustering algorithm that can find arbitrarily shaped clusters.",
    difficulty: "hard",
    concepts: ["DBSCAN", "density-based", "eps", "min_samples", "noise points"],
    hints: [
      "DBSCAN finds dense regions separated by sparse regions",
      "eps: maximum distance between two samples to be considered neighbors",
      "min_samples: minimum points needed to form a dense region",
      "Labels of -1 indicate noise points",
    ],
    docsUrl: ["https://scikit-learn.org/stable/modules/clustering.html#dbscan"],
    code: `from sklearn.datasets import make_moons, make_blobs
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

# Create non-convex clusters (moons)
X, y_true = make_moons(n_samples=300, noise=0.1, random_state=42)
X = StandardScaler().fit_transform(X)

print("Comparing K-Means vs DBSCAN on 'moons' data")
print("-" * 45)

# K-Means (struggles with non-convex shapes)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

# TODO: Create DBSCAN with eps=0.3 and min_samples=5
dbscan = None  # <- DBSCAN(eps=0.3, min_samples=5)

# TODO: Fit and get labels
dbscan_labels = None  # <- dbscan.fit_predict(X)

# Count clusters and noise
n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

print(f"DBSCAN found {n_clusters} clusters and {n_noise} noise points")
print()

# Visualize comparison
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='tab10')
plt.title('True Labels')

plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='tab10')
plt.title('K-Means (k=2)')

plt.subplot(1, 3, 3)
colors = ['gray' if l == -1 else plt.cm.tab10(l) for l in dbscan_labels]
plt.scatter(X[:, 0], X[:, 1], c=colors)
plt.title(f'DBSCAN ({n_clusters} clusters, {n_noise} noise)')

plt.tight_layout()
plt.show()

# Effect of eps parameter
print("Effect of eps parameter:")
for eps in [0.1, 0.2, 0.3, 0.5, 1.0]:
    db = DBSCAN(eps=eps, min_samples=5)
    labels = db.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"  eps={eps}: clusters={n_clusters}, noise={n_noise}")

result = f"DBSCAN: {n_clusters} clusters, {n_noise} noise points"
result`,
    solution: `from sklearn.datasets import make_moons, make_blobs
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

# Create non-convex clusters (moons)
X, y_true = make_moons(n_samples=300, noise=0.1, random_state=42)
X = StandardScaler().fit_transform(X)

print("Comparing K-Means vs DBSCAN on 'moons' data")
print("-" * 45)

# K-Means (struggles with non-convex shapes)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

# Create DBSCAN with eps=0.3 and min_samples=5
dbscan = DBSCAN(eps=0.3, min_samples=5)

# Fit and get labels
dbscan_labels = dbscan.fit_predict(X)

# Count clusters and noise
n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

print(f"DBSCAN found {n_clusters} clusters and {n_noise} noise points")
print()

# Visualize comparison
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='tab10')
plt.title('True Labels')

plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='tab10')
plt.title('K-Means (k=2)')

plt.subplot(1, 3, 3)
colors = ['gray' if l == -1 else plt.cm.tab10(l) for l in dbscan_labels]
plt.scatter(X[:, 0], X[:, 1], c=colors)
plt.title(f'DBSCAN ({n_clusters} clusters, {n_noise} noise)')

plt.tight_layout()
plt.show()

# Effect of eps parameter
print("Effect of eps parameter:")
for eps in [0.1, 0.2, 0.3, 0.5, 1.0]:
    db = DBSCAN(eps=eps, min_samples=5)
    labels = db.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"  eps={eps}: clusters={n_clusters}, noise={n_noise}")

result = f"DBSCAN: {n_clusters} clusters, {n_noise} noise points"
result`,
  },
  {
    id: "feature-selection",
    type: "course",
    phase: 3,
    order: 8,
    title: "Feature Selection Techniques",
    description:
      "Learn various feature selection methods to improve model performance and reduce complexity.",
    difficulty: "hard",
    concepts: ["SelectKBest", "RFE", "feature selection", "chi2", "mutual_info"],
    hints: [
      "Filter methods: score features independently (fast)",
      "Wrapper methods: use model performance to select features (slower, often better)",
      "Embedded methods: feature selection during training (e.g., L1 regularization)",
    ],
    docsUrl: ["https://scikit-learn.org/stable/modules/feature_selection.html"],
    code: `from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load data
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

print(f"Original features: {X.shape[1]}")
print()

# Method 1: SelectKBest with mutual information
# TODO: Create SelectKBest with k=10
selector = None  # <- SelectKBest(score_func=mutual_info_classif, k=10)
X_selected = None  # <- selector.fit_transform(X, y)

# Get selected feature names
mask = selector.get_support()
selected_features = [cancer.feature_names[i] for i in range(len(mask)) if mask[i]]

print("SelectKBest (k=10) selected features:")
for f in selected_features[:5]:
    print(f"  - {f}")
print(f"  ... and {len(selected_features) - 5} more")
print()

# Method 2: Recursive Feature Elimination (RFE)
rfe = RFE(estimator=LogisticRegression(max_iter=5000), n_features_to_select=10)
X_rfe = rfe.fit_transform(X, y)

rfe_mask = rfe.support_
rfe_features = [cancer.feature_names[i] for i in range(len(rfe_mask)) if rfe_mask[i]]

print("RFE (n=10) selected features:")
for f in rfe_features[:5]:
    print(f"  - {f}")
print(f"  ... and {len(rfe_features) - 5} more")
print()

# Compare performance
clf = LogisticRegression(max_iter=5000, random_state=42)

scores_all = cross_val_score(clf, X, y, cv=5)
scores_kbest = cross_val_score(clf, X_selected, y, cv=5)
scores_rfe = cross_val_score(clf, X_rfe, y, cv=5)

print("Cross-validation scores:")
print(f"  All {X.shape[1]} features: {scores_all.mean():.3f}")
print(f"  SelectKBest (10):   {scores_kbest.mean():.3f}")
print(f"  RFE (10):           {scores_rfe.mean():.3f}")

result = f"Best with {len(selected_features)} features"
result`,
    solution: `from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load data
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

print(f"Original features: {X.shape[1]}")
print()

# Method 1: SelectKBest with mutual information
# Create SelectKBest with k=10
selector = SelectKBest(score_func=mutual_info_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Get selected feature names
mask = selector.get_support()
selected_features = [cancer.feature_names[i] for i in range(len(mask)) if mask[i]]

print("SelectKBest (k=10) selected features:")
for f in selected_features[:5]:
    print(f"  - {f}")
print(f"  ... and {len(selected_features) - 5} more")
print()

# Method 2: Recursive Feature Elimination (RFE)
rfe = RFE(estimator=LogisticRegression(max_iter=5000), n_features_to_select=10)
X_rfe = rfe.fit_transform(X, y)

rfe_mask = rfe.support_
rfe_features = [cancer.feature_names[i] for i in range(len(rfe_mask)) if rfe_mask[i]]

print("RFE (n=10) selected features:")
for f in rfe_features[:5]:
    print(f"  - {f}")
print(f"  ... and {len(rfe_features) - 5} more")
print()

# Compare performance
clf = LogisticRegression(max_iter=5000, random_state=42)

scores_all = cross_val_score(clf, X, y, cv=5)
scores_kbest = cross_val_score(clf, X_selected, y, cv=5)
scores_rfe = cross_val_score(clf, X_rfe, y, cv=5)

print("Cross-validation scores:")
print(f"  All {X.shape[1]} features: {scores_all.mean():.3f}")
print(f"  SelectKBest (10):   {scores_kbest.mean():.3f}")
print(f"  RFE (10):           {scores_rfe.mean():.3f}")

result = f"Best with {len(selected_features)} features"
result`,
  },
  {
    id: "imbalanced-data",
    type: "course",
    phase: 3,
    order: 9,
    title: "Handling Imbalanced Data",
    description:
      "Learn techniques for handling datasets with unequal class distributions.",
    difficulty: "hard",
    concepts: ["class_weight", "imbalanced", "stratified sampling", "balanced accuracy"],
    hints: [
      "class_weight='balanced' adjusts weights inversely proportional to class frequencies",
      "Stratified CV ensures each fold has the same class distribution",
      "Use balanced_accuracy_score for imbalanced evaluation",
    ],
    docsUrl: ["https://scikit-learn.org/stable/modules/model_evaluation.html#balanced-accuracy-score"],
    code: `from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report
import numpy as np

# Create imbalanced dataset (1:9 ratio)
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=10,
    n_classes=2, weights=[0.9, 0.1], random_state=42
)

print("Class distribution:")
print(f"  Class 0: {(y == 0).sum()} samples")
print(f"  Class 1: {(y == 1).sum()} samples")
print(f"  Ratio: {(y == 0).sum() / (y == 1).sum():.1f}:1")
print()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standard model (ignores imbalance)
clf_standard = LogisticRegression(random_state=42)
clf_standard.fit(X_train, y_train)
y_pred_standard = clf_standard.predict(X_test)

# TODO: Create model with class_weight='balanced'
clf_balanced = None  # <- LogisticRegression(class_weight='balanced', random_state=42)
None  # <- fit on training data
y_pred_balanced = clf_balanced.predict(X_test)

print("Standard Model:")
print(f"  Accuracy: {clf_standard.score(X_test, y_test):.3f}")
print(f"  Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred_standard):.3f}")
print()

print("Balanced Model:")
print(f"  Accuracy: {clf_balanced.score(X_test, y_test):.3f}")
print(f"  Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred_balanced):.3f}")
print()

# Detailed comparison
print("Classification Report (Balanced Model):")
print(classification_report(y_test, y_pred_balanced))

# Stratified cross-validation
print("Stratified 5-Fold CV:")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_standard = cross_val_score(clf_standard, X, y, cv=cv, scoring='balanced_accuracy')
scores_balanced = cross_val_score(clf_balanced, X, y, cv=cv, scoring='balanced_accuracy')
print(f"  Standard: {scores_standard.mean():.3f}")
print(f"  Balanced: {scores_balanced.mean():.3f}")

result = f"Balanced accuracy improved: {balanced_accuracy_score(y_test, y_pred_balanced):.3f}"
result`,
    solution: `from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report
import numpy as np

# Create imbalanced dataset (1:9 ratio)
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=10,
    n_classes=2, weights=[0.9, 0.1], random_state=42
)

print("Class distribution:")
print(f"  Class 0: {(y == 0).sum()} samples")
print(f"  Class 1: {(y == 1).sum()} samples")
print(f"  Ratio: {(y == 0).sum() / (y == 1).sum():.1f}:1")
print()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standard model (ignores imbalance)
clf_standard = LogisticRegression(random_state=42)
clf_standard.fit(X_train, y_train)
y_pred_standard = clf_standard.predict(X_test)

# Create model with class_weight='balanced'
clf_balanced = LogisticRegression(class_weight='balanced', random_state=42)
clf_balanced.fit(X_train, y_train)
y_pred_balanced = clf_balanced.predict(X_test)

print("Standard Model:")
print(f"  Accuracy: {clf_standard.score(X_test, y_test):.3f}")
print(f"  Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred_standard):.3f}")
print()

print("Balanced Model:")
print(f"  Accuracy: {clf_balanced.score(X_test, y_test):.3f}")
print(f"  Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred_balanced):.3f}")
print()

# Detailed comparison
print("Classification Report (Balanced Model):")
print(classification_report(y_test, y_pred_balanced))

# Stratified cross-validation
print("Stratified 5-Fold CV:")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_standard = cross_val_score(clf_standard, X, y, cv=cv, scoring='balanced_accuracy')
scores_balanced = cross_val_score(clf_balanced, X, y, cv=cv, scoring='balanced_accuracy')
print(f"  Standard: {scores_standard.mean():.3f}")
print(f"  Balanced: {scores_balanced.mean():.3f}")

result = f"Balanced accuracy improved: {balanced_accuracy_score(y_test, y_pred_balanced):.3f}"
result`,
  },
  {
    id: "complete-ml-pipeline",
    type: "course",
    phase: 3,
    order: 10,
    title: "Complete ML Pipeline Project",
    description:
      "Build a complete machine learning pipeline from data preprocessing to model evaluation.",
    difficulty: "hard",
    concepts: ["full pipeline", "production ML", "best practices", "model selection"],
    hints: [
      "Follow the workflow: preprocess -> feature engineer -> model select -> tune -> evaluate",
      "Use pipelines to prevent data leakage",
      "Compare multiple models before selecting the best one",
    ],
    docsUrl: [
      "https://scikit-learn.org/stable/getting_started.html",
      "https://scikit-learn.org/stable/modules/compose.html#pipeline"
    ],
    code: `from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# 1. Load and split data
print("=== Step 1: Load Data ===")
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42, stratify=wine.target
)
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features: {wine.feature_names}")
print()

# 2. Define candidate pipelines
print("=== Step 2: Define Candidate Pipelines ===")
pipelines = {
    'Logistic': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'SVM': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(random_state=42))
    ]),
    'RandomForest': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(random_state=42))
    ]),
    'GradientBoosting': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(random_state=42))
    ])
}
print(f"Candidates: {list(pipelines.keys())}")
print()

# 3. Compare models with cross-validation
print("=== Step 3: Model Comparison (5-fold CV) ===")
# TODO: Run cross-validation for each pipeline
best_score = 0
best_name = None

for name, pipe in pipelines.items():
    scores = None  # <- cross_val_score(pipe, X_train, y_train, cv=5)
    print(f"  {name:17}: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")
    if scores.mean() > best_score:
        best_score = scores.mean()
        best_name = name

print(f"\\nBest candidate: {best_name}")
print()

# 4. Tune the best model
print("=== Step 4: Hyperparameter Tuning ===")
# Fine-tune the best model (using SVM as example)
param_grid = {
    'clf__C': [0.1, 1, 10],
    'clf__kernel': ['linear', 'rbf']
}

# TODO: Create and fit GridSearchCV
grid_search = GridSearchCV(
    pipelines['SVM'], param_grid, cv=5, scoring='accuracy'
)
None  # <- grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")
print()

# 5. Final evaluation on test set
print("=== Step 5: Final Evaluation ===")
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print(f"Test accuracy: {best_model.score(X_test, y_test):.3f}")
print()
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=wine.target_names))

result = f"Final model accuracy: {best_model.score(X_test, y_test):.3f}"
result`,
    solution: `from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# 1. Load and split data
print("=== Step 1: Load Data ===")
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42, stratify=wine.target
)
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features: {wine.feature_names}")
print()

# 2. Define candidate pipelines
print("=== Step 2: Define Candidate Pipelines ===")
pipelines = {
    'Logistic': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'SVM': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(random_state=42))
    ]),
    'RandomForest': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(random_state=42))
    ]),
    'GradientBoosting': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(random_state=42))
    ])
}
print(f"Candidates: {list(pipelines.keys())}")
print()

# 3. Compare models with cross-validation
print("=== Step 3: Model Comparison (5-fold CV) ===")
# Run cross-validation for each pipeline
best_score = 0
best_name = None

for name, pipe in pipelines.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=5)
    print(f"  {name:17}: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")
    if scores.mean() > best_score:
        best_score = scores.mean()
        best_name = name

print(f"\\nBest candidate: {best_name}")
print()

# 4. Tune the best model
print("=== Step 4: Hyperparameter Tuning ===")
# Fine-tune the best model (using SVM as example)
param_grid = {
    'clf__C': [0.1, 1, 10],
    'clf__kernel': ['linear', 'rbf']
}

# Create and fit GridSearchCV
grid_search = GridSearchCV(
    pipelines['SVM'], param_grid, cv=5, scoring='accuracy'
)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")
print()

# 5. Final evaluation on test set
print("=== Step 5: Final Evaluation ===")
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print(f"Test accuracy: {best_model.score(X_test, y_test):.3f}")
print()
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=wine.target_names))

result = f"Final model accuracy: {best_model.score(X_test, y_test):.3f}"
result`,
  },
];
