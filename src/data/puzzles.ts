export interface Puzzle {
  id: string;
  title: string;
  description: string;
  context?: string;
  code: string;
  solution?: string;
  difficulty?: "easy" | "medium" | "hard";
  docsUrl?: string[];
}

export const puzzles: Puzzle[] = [
  {
    id: "train-test-split",
    title: "Basic Train/Test Split",
    description:
      "Split the iris dataset into training and test sets with 80% for training and 20% for testing. Use random_state=42 for reproducibility.",
    context:
      "Train/test splitting is fundamental to ML - it lets you evaluate how well your model generalizes to unseen data. Without this, you might build a model that memorizes training data but fails on new examples. The 80/20 split is a common starting point.",
    difficulty: "easy",
    docsUrl: ["https://scikit-learn.org/stable/modules/cross_validation.html"],
    code: `from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

print(f"Total samples: {len(X)}")

# TODO: Split the data into X_train, X_test, y_train, y_test
# Use test_size=0.2 and random_state=42
X_train, X_test, y_train, y_test = None, None, None, None  # <- modify this line

# Verify the split
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

result = f"Train: {len(X_train)}, Test: {len(X_test)}"
result`,
    solution: `from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

print(f"Total samples: {len(X)}")

# Split the data into X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verify the split
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

result = f"Train: {len(X_train)}, Test: {len(X_test)}"
result`,
  },
  {
    id: "fit-predict-knn",
    title: "KNN Fit and Predict",
    description:
      "Train a K-Nearest Neighbors classifier on the iris dataset and make predictions on the test set.",
    context:
      "The fit/predict pattern is the core sklearn API. KNN is intuitive - it classifies based on the majority vote of k nearest neighbors. It's often used as a baseline and works well for small datasets with clear cluster separation.",
    difficulty: "easy",
    docsUrl: ["https://scikit-learn.org/stable/modules/neighbors.html"],
    code: `from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load and split the data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# TODO: Create a KNN classifier with n_neighbors=3, fit it, and predict
knn = None  # <- create the classifier
# <- fit the classifier on training data
y_pred = None  # <- make predictions on X_test

# Check results
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Predictions: {y_pred[:10]}")
print(f"Actual:      {y_test[:10]}")
print(f"Accuracy: {accuracy:.2%}")

result = f"Accuracy: {accuracy:.2%}"
result`,
    solution: `from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load and split the data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Create a KNN classifier with n_neighbors=3, fit it, and predict
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Check results
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Predictions: {y_pred[:10]}")
print(f"Actual:      {y_test[:10]}")
print(f"Accuracy: {accuracy:.2%}")

result = f"Accuracy: {accuracy:.2%}"
result`,
  },
  {
    id: "confusion-matrix",
    title: "Build a Confusion Matrix",
    description:
      "Create and display a confusion matrix for a classifier's predictions on the iris dataset.",
    context:
      "Accuracy alone can be misleading, especially with imbalanced classes. A confusion matrix shows exactly which classes are being confused with which - essential for understanding model behavior and debugging classification problems.",
    difficulty: "easy",
    docsUrl: ["https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix"],
    code: `from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import numpy as np

# Train a simple classifier
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# TODO: Create the confusion matrix
cm = None  # <- create confusion matrix using y_test and y_pred

print("Confusion Matrix:")
print(cm)
print(f"\\nClass names: {iris.target_names}")

result = cm
result`,
    solution: `from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import numpy as np

# Train a simple classifier
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
print(f"\\nClass names: {iris.target_names}")

result = cm
result`,
  },
  {
    id: "standardize-features",
    title: "Standardize Features",
    description:
      "Use StandardScaler to standardize features so they have zero mean and unit variance.",
    context:
      "Many ML algorithms (SVM, KNN, neural networks) are sensitive to feature scales. If one feature ranges 0-1 and another 0-1000, the larger one dominates distance calculations. Standardization puts all features on equal footing.",
    difficulty: "easy",
    docsUrl: ["https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling"],
    code: `from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load wine dataset
wine = load_wine()
X = wine.data

print("Before scaling:")
print(f"Feature 0 - Mean: {X[:, 0].mean():.2f}, Std: {X[:, 0].std():.2f}")
print(f"Feature 1 - Mean: {X[:, 1].mean():.2f}, Std: {X[:, 1].std():.2f}")

# TODO: Create a StandardScaler, fit it, and transform the data
scaler = None  # <- create the scaler
X_scaled = None  # <- fit and transform the data

print("\\nAfter scaling:")
print(f"Feature 0 - Mean: {X_scaled[:, 0].mean():.2f}, Std: {X_scaled[:, 0].std():.2f}")
print(f"Feature 1 - Mean: {X_scaled[:, 1].mean():.2f}, Std: {X_scaled[:, 1].std():.2f}")

result = f"Scaled mean: {X_scaled.mean():.4f}"
result`,
    solution: `from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load wine dataset
wine = load_wine()
X = wine.data

print("Before scaling:")
print(f"Feature 0 - Mean: {X[:, 0].mean():.2f}, Std: {X[:, 0].std():.2f}")
print(f"Feature 1 - Mean: {X[:, 1].mean():.2f}, Std: {X[:, 1].std():.2f}")

# Create a StandardScaler, fit it, and transform the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\\nAfter scaling:")
print(f"Feature 0 - Mean: {X_scaled[:, 0].mean():.2f}, Std: {X_scaled[:, 0].std():.2f}")
print(f"Feature 1 - Mean: {X_scaled[:, 1].mean():.2f}, Std: {X_scaled[:, 1].std():.2f}")

result = f"Scaled mean: {X_scaled.mean():.4f}"
result`,
  },
  {
    id: "cross-validation",
    title: "Cross-Validation Score",
    description:
      "Use 5-fold cross-validation to evaluate a Random Forest classifier on the digits dataset.",
    context:
      "A single train/test split can give misleading results depending on how the data happens to be divided. Cross-validation provides a more robust estimate by training and testing on multiple different splits, giving you both a mean score and variance.",
    difficulty: "medium",
    docsUrl: ["https://scikit-learn.org/stable/modules/cross_validation.html"],
    code: `from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# Load digits dataset
digits = load_digits()
X, y = digits.data, digits.target

print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")

# TODO: Create a RandomForestClassifier and get cross-validation scores
clf = RandomForestClassifier(n_estimators=50, random_state=42)
scores = None  # <- use cross_val_score with cv=5

print(f"\\nCV Scores: {scores}")
print(f"Mean accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

result = f"Mean CV accuracy: {scores.mean():.3f}"
result`,
    solution: `from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# Load digits dataset
digits = load_digits()
X, y = digits.data, digits.target

print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")

# Create a RandomForestClassifier and get cross-validation scores
clf = RandomForestClassifier(n_estimators=50, random_state=42)
scores = cross_val_score(clf, X, y, cv=5)

print(f"\\nCV Scores: {scores}")
print(f"Mean accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

result = f"Mean CV accuracy: {scores.mean():.3f}"
result`,
  },
  {
    id: "pipeline-basics",
    title: "Build a Simple Pipeline",
    description:
      "Create a pipeline that first scales the data and then applies logistic regression.",
    context:
      "Pipelines prevent data leakage by ensuring preprocessing is fit only on training data. They also make your code cleaner and your workflow reproducible - essential for production ML systems where you need consistent preprocessing.",
    difficulty: "medium",
    docsUrl: ["https://scikit-learn.org/stable/modules/compose.html#pipeline"],
    code: `from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Load data
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42
)

# TODO: Create a pipeline with StandardScaler and LogisticRegression
pipe = None  # <- create Pipeline with steps: [('scaler', ...), ('classifier', ...)]

# Fit and evaluate
pipe.fit(X_train, y_train)
accuracy = pipe.score(X_test, y_test)

print(f"Pipeline steps: {[step[0] for step in pipe.steps]}")
print(f"Test accuracy: {accuracy:.3f}")

result = f"Pipeline accuracy: {accuracy:.3f}"
result`,
    solution: `from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Load data
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42
)

# Create a pipeline with StandardScaler and LogisticRegression
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=5000, random_state=42))
])

# Fit and evaluate
pipe.fit(X_train, y_train)
accuracy = pipe.score(X_test, y_test)

print(f"Pipeline steps: {[step[0] for step in pipe.steps]}")
print(f"Test accuracy: {accuracy:.3f}")

result = f"Pipeline accuracy: {accuracy:.3f}"
result`,
  },
  {
    id: "grid-search",
    title: "Grid Search Hyperparameters",
    description:
      "Use GridSearchCV to find the best hyperparameters for an SVM classifier.",
    context:
      "Hyperparameter tuning can dramatically improve model performance. GridSearchCV automates this by systematically trying all combinations and using cross-validation to find the best settings - much more reliable than manual tuning.",
    difficulty: "medium",
    docsUrl: ["https://scikit-learn.org/stable/modules/grid_search.html#exhaustive-grid-search"],
    code: `from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

# TODO: Create GridSearchCV with SVC and the param_grid, cv=3
grid_search = None  # <- create GridSearchCV

# Fit and get results
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")
print(f"Test score: {grid_search.score(X_test, y_test):.3f}")

result = f"Best params: {grid_search.best_params_}"
result`,
    solution: `from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

# Create GridSearchCV with SVC and the param_grid, cv=3
grid_search = GridSearchCV(SVC(), param_grid, cv=3)

# Fit and get results
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")
print(f"Test score: {grid_search.score(X_test, y_test):.3f}")

result = f"Best params: {grid_search.best_params_}"
result`,
  },
  {
    id: "feature-importance",
    title: "Extract Feature Importance",
    description:
      "Train a Random Forest and extract the feature importances for the wine dataset.",
    context:
      "Feature importance helps you understand which inputs matter most for predictions. This is crucial for model interpretation, feature selection, and explaining predictions to stakeholders - key for building trust in ML systems.",
    difficulty: "medium",
    docsUrl: ["https://scikit-learn.org/stable/modules/ensemble.html#feature-importance-evaluation"],
    code: `from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load wine dataset
wine = load_wine()
X, y = wine.data, wine.target

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# TODO: Get feature importances and find the top 3 most important features
importances = None  # <- get feature_importances_ from the trained model
top_3_idx = None  # <- get indices of top 3 features using np.argsort

print("Top 3 most important features:")
for idx in top_3_idx:
    print(f"  {wine.feature_names[idx]}: {importances[idx]:.4f}")

result = f"Top feature: {wine.feature_names[top_3_idx[0]]}"
result`,
    solution: `from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load wine dataset
wine = load_wine()
X, y = wine.data, wine.target

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Get feature importances and find the top 3 most important features
importances = rf.feature_importances_
top_3_idx = np.argsort(importances)[::-1][:3]

print("Top 3 most important features:")
for idx in top_3_idx:
    print(f"  {wine.feature_names[idx]}: {importances[idx]:.4f}")

result = f"Top feature: {wine.feature_names[top_3_idx[0]]}"
result`,
  },
  {
    id: "pca-dimensionality",
    title: "PCA Dimensionality Reduction",
    description:
      "Use PCA to reduce the digits dataset from 64 dimensions to 2 and visualize explained variance.",
    context:
      "High-dimensional data is hard to visualize and can cause overfitting. PCA finds the most informative directions in your data, letting you reduce dimensions while retaining most of the variance. It's also great for visualization and denoising.",
    difficulty: "medium",
    docsUrl: ["https://scikit-learn.org/stable/modules/decomposition.html#pca"],
    code: `from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import numpy as np

# Load digits dataset (64 features)
digits = load_digits()
X = digits.data

print(f"Original shape: {X.shape}")

# TODO: Create PCA with n_components=2 and fit_transform the data
pca = None  # <- create PCA
X_reduced = None  # <- fit and transform

print(f"Reduced shape: {X_reduced.shape}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")

result = f"Variance explained: {sum(pca.explained_variance_ratio_):.2%}"
result`,
    solution: `from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import numpy as np

# Load digits dataset (64 features)
digits = load_digits()
X = digits.data

print(f"Original shape: {X.shape}")

# Create PCA with n_components=2 and fit_transform the data
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

print(f"Reduced shape: {X_reduced.shape}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")

result = f"Variance explained: {sum(pca.explained_variance_ratio_):.2%}"
result`,
  },
  {
    id: "kmeans-clustering",
    title: "K-Means Clustering",
    description:
      "Apply K-Means clustering to the iris dataset and compare cluster labels with true labels.",
    context:
      "Clustering finds natural groupings in data without labels - useful for customer segmentation, anomaly detection, and exploratory analysis. K-Means is fast and intuitive, making it a go-to algorithm for clustering tasks.",
    difficulty: "medium",
    docsUrl: ["https://scikit-learn.org/stable/modules/clustering.html#k-means"],
    code: `from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import numpy as np

# Load iris dataset
iris = load_iris()
X = iris.data
y_true = iris.target

# TODO: Create KMeans with n_clusters=3 and random_state=42, then fit and predict
kmeans = None  # <- create KMeans
y_pred = None  # <- fit_predict on X

# Compare clusters to true labels
ari = adjusted_rand_score(y_true, y_pred)

print(f"Cluster labels: {np.unique(y_pred)}")
print(f"Cluster sizes: {[sum(y_pred == i) for i in range(3)]}")
print(f"Adjusted Rand Index: {ari:.3f}")
print(f"Inertia: {kmeans.inertia_:.2f}")

result = f"ARI: {ari:.3f}"
result`,
    solution: `from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import numpy as np

# Load iris dataset
iris = load_iris()
X = iris.data
y_true = iris.target

# Create KMeans with n_clusters=3 and random_state=42, then fit and predict
kmeans = KMeans(n_clusters=3, random_state=42)
y_pred = kmeans.fit_predict(X)

# Compare clusters to true labels
ari = adjusted_rand_score(y_true, y_pred)

print(f"Cluster labels: {np.unique(y_pred)}")
print(f"Cluster sizes: {[sum(y_pred == i) for i in range(3)]}")
print(f"Adjusted Rand Index: {ari:.3f}")
print(f"Inertia: {kmeans.inertia_:.2f}")

result = f"ARI: {ari:.3f}"
result`,
  },
  {
    id: "roc-curve",
    title: "Plot ROC Curve",
    description:
      "Train a classifier and compute the ROC curve and AUC score for binary classification.",
    context:
      "ROC curves show the trade-off between true positive rate and false positive rate at different classification thresholds. AUC provides a single number summarizing this trade-off - essential for comparing classifiers and choosing optimal thresholds.",
    difficulty: "hard",
    docsUrl: ["https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics"],
    code: `from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Load binary classification data
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.3, random_state=42
)

# Train classifier
clf = LogisticRegression(max_iter=5000, random_state=42)
clf.fit(X_train, y_train)

# TODO: Get probability predictions and compute ROC curve
y_prob = None  # <- get probability for positive class using predict_proba
fpr, tpr, _ = None, None, None  # <- compute roc_curve
roc_auc = None  # <- compute AUC score

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.show()

result = f"AUC: {roc_auc:.3f}"
result`,
    solution: `from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Load binary classification data
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.3, random_state=42
)

# Train classifier
clf = LogisticRegression(max_iter=5000, random_state=42)
clf.fit(X_train, y_train)

# Get probability predictions and compute ROC curve
y_prob = clf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.show()

result = f"AUC: {roc_auc:.3f}"
result`,
  },
  {
    id: "voting-classifier",
    title: "Ensemble Voting Classifier",
    description:
      "Create a voting classifier that combines multiple different classifiers.",
    context:
      "Ensemble methods combine multiple models to get better predictions than any single model. Voting classifiers are simple but effective - they leverage the wisdom of crowds, where diverse models often make different mistakes that cancel out.",
    difficulty: "hard",
    docsUrl: ["https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier"],
    code: `from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Define individual classifiers
clf1 = LogisticRegression(max_iter=1000, random_state=42)
clf2 = RandomForestClassifier(n_estimators=50, random_state=42)
clf3 = SVC(probability=True, random_state=42)

# TODO: Create a VotingClassifier with soft voting
voting_clf = None  # <- create VotingClassifier with estimators and voting='soft'

# Compare individual vs ensemble performance
print("Cross-validation scores (5-fold):")
for clf, name in [(clf1, 'Logistic'), (clf2, 'RandomForest'), (clf3, 'SVC'), (voting_clf, 'Voting')]:
    scores = cross_val_score(clf, X, y, cv=5)
    print(f"  {name}: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

result = f"Voting ensemble built with 3 classifiers"
result`,
    solution: `from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Define individual classifiers
clf1 = LogisticRegression(max_iter=1000, random_state=42)
clf2 = RandomForestClassifier(n_estimators=50, random_state=42)
clf3 = SVC(probability=True, random_state=42)

# Create a VotingClassifier with soft voting
voting_clf = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('svc', clf3)],
    voting='soft'
)

# Compare individual vs ensemble performance
print("Cross-validation scores (5-fold):")
for clf, name in [(clf1, 'Logistic'), (clf2, 'RandomForest'), (clf3, 'SVC'), (voting_clf, 'Voting')]:
    scores = cross_val_score(clf, X, y, cv=5)
    print(f"  {name}: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

result = f"Voting ensemble built with 3 classifiers"
result`,
  },
  {
    id: "precision-recall",
    title: "Precision-Recall Trade-off",
    description:
      "Calculate precision, recall, and F1-score for different classification thresholds.",
    context:
      "In real applications, the costs of false positives vs false negatives are often different (spam detection vs cancer diagnosis). Understanding precision-recall trade-offs lets you tune your classifier for your specific business needs.",
    difficulty: "hard",
    docsUrl: ["https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-and-f-measures"],
    code: `from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# Load data
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.3, random_state=42
)

# Train classifier
clf = LogisticRegression(max_iter=5000, random_state=42)
clf.fit(X_train, y_train)

# Get probabilities
y_prob = clf.predict_proba(X_test)[:, 1]

# TODO: Calculate metrics for different thresholds
thresholds = [0.3, 0.5, 0.7]
print("Threshold | Precision | Recall | F1-Score")
print("-" * 45)

for thresh in thresholds:
    y_pred = (y_prob >= thresh).astype(int)  # <- convert probabilities to predictions
    precision = None  # <- calculate precision
    recall = None  # <- calculate recall
    f1 = None  # <- calculate f1_score
    print(f"   {thresh}     |   {precision:.3f}   | {recall:.3f}  |  {f1:.3f}")

result = "Precision-recall analysis complete"
result`,
    solution: `from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# Load data
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.3, random_state=42
)

# Train classifier
clf = LogisticRegression(max_iter=5000, random_state=42)
clf.fit(X_train, y_train)

# Get probabilities
y_prob = clf.predict_proba(X_test)[:, 1]

# Calculate metrics for different thresholds
thresholds = [0.3, 0.5, 0.7]
print("Threshold | Precision | Recall | F1-Score")
print("-" * 45)

for thresh in thresholds:
    y_pred = (y_prob >= thresh).astype(int)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"   {thresh}     |   {precision:.3f}   | {recall:.3f}  |  {f1:.3f}")

result = "Precision-recall analysis complete"
result`,
  },
  {
    id: "column-transformer",
    title: "Mixed Data Types Pipeline",
    description:
      "Use ColumnTransformer to handle numerical and categorical features differently in a pipeline.",
    context:
      "Real-world datasets have mixed types - numbers, categories, text. ColumnTransformer lets you apply different preprocessing to different columns, then combine them. This is how production ML pipelines handle messy real data.",
    difficulty: "hard",
    docsUrl: ["https://scikit-learn.org/stable/modules/compose.html#columntransformer-for-heterogeneous-data"],
    code: `from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

# Create a mixed dataset
np.random.seed(42)
df = pd.DataFrame({
    'age': np.random.randint(20, 60, 100),
    'income': np.random.randint(30000, 100000, 100),
    'education': np.random.choice(['high_school', 'bachelors', 'masters'], 100),
    'city': np.random.choice(['NYC', 'LA', 'Chicago'], 100),
    'target': np.random.randint(0, 2, 100)
})

X = df.drop('target', axis=1)
y = df['target']

print("Data preview:")
print(X.head())

# Define column types
numeric_features = ['age', 'income']
categorical_features = ['education', 'city']

# TODO: Create a ColumnTransformer with:
# - StandardScaler for numeric features (name it 'num')
# - OneHotEncoder for categorical features (name it 'cat')
preprocessor = None  # <- create ColumnTransformer

# Create full pipeline
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))
])

# Fit and score
pipe.fit(X, y)
print(f"\\nPipeline score: {pipe.score(X, y):.3f}")

result = "ColumnTransformer pipeline complete"
result`,
    solution: `from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

# Create a mixed dataset
np.random.seed(42)
df = pd.DataFrame({
    'age': np.random.randint(20, 60, 100),
    'income': np.random.randint(30000, 100000, 100),
    'education': np.random.choice(['high_school', 'bachelors', 'masters'], 100),
    'city': np.random.choice(['NYC', 'LA', 'Chicago'], 100),
    'target': np.random.randint(0, 2, 100)
})

X = df.drop('target', axis=1)
y = df['target']

print("Data preview:")
print(X.head())

# Define column types
numeric_features = ['age', 'income']
categorical_features = ['education', 'city']

# Create a ColumnTransformer with:
# - StandardScaler for numeric features (name it 'num')
# - OneHotEncoder for categorical features (name it 'cat')
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Create full pipeline
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))
])

# Fit and score
pipe.fit(X, y)
print(f"\\nPipeline score: {pipe.score(X, y):.3f}")

result = "ColumnTransformer pipeline complete"
result`,
  },
];

export function getPuzzleById(id: string): Puzzle | undefined {
  return puzzles.find((p) => p.id === id);
}
