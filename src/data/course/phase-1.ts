import type { CourseExercise } from "../types";

export const phase1Exercises: CourseExercise[] = [
  {
    id: "loading-datasets",
    type: "course",
    phase: 1,
    order: 1,
    title: "Loading Datasets",
    description:
      "Learn how to load built-in datasets from sklearn.datasets. These datasets return numpy arrays, which we'll convert to Polars DataFrames for easier exploration.",
    difficulty: "easy",
    concepts: ["datasets", "load_iris", "numpy arrays", "polars conversion"],
    hints: [
      "Use load_iris() to get the iris dataset",
      "The returned object has .data (numpy array), .target, and .feature_names attributes",
      "Convert to Polars with pl.DataFrame(data, schema=column_names)",
    ],
    docsUrl: ["https://scikit-learn.org/stable/datasets/toy_dataset.html"],
    code: `from sklearn.datasets import load_iris, load_digits, load_wine
import polars as pl

# Load the famous iris dataset
# sklearn returns numpy arrays, not DataFrames
iris = load_iris()

# Explore what sklearn gives us (numpy arrays)
print("Type of iris.data:", type(iris.data))
print("Data shape:", iris.data.shape)
print("Feature names:", iris.feature_names)
print("Target names (classes):", iris.target_names)
print()

# TODO: Convert numpy arrays to a Polars DataFrame for easier exploration
# Use pl.DataFrame(data, schema=column_names)
df = pl.DataFrame(None, schema=None)  # <- iris.data, iris.feature_names

# TODO: Add the target column (species as integer: 0, 1, 2)
df = df.with_columns(None)  # <- pl.Series("species", iris.target)

print("As Polars DataFrame:")
df.glimpse()
print()

# Now we have a nice DataFrame for EDA!
print(f"Shape: {df.shape}")

result = f"Loaded iris with {df.shape[0]} samples and {df.shape[1]} columns"
result`,
    solution: `from sklearn.datasets import load_iris, load_digits, load_wine
import polars as pl

# Load the famous iris dataset
# sklearn returns numpy arrays, not DataFrames
iris = load_iris()

# Explore what sklearn gives us (numpy arrays)
print("Type of iris.data:", type(iris.data))
print("Data shape:", iris.data.shape)
print("Feature names:", iris.feature_names)
print("Target names (classes):", iris.target_names)
print()

# Convert numpy arrays to a Polars DataFrame for easier exploration
df = pl.DataFrame(iris.data, schema=iris.feature_names)

# Add the target column (species as integer: 0, 1, 2)
df = df.with_columns(pl.Series("species", iris.target))

print("As Polars DataFrame:")
df.glimpse()
print()

# Now we have a nice DataFrame for EDA!
print(f"Shape: {df.shape}")

result = f"Loaded iris with {df.shape[0]} samples and {df.shape[1]} columns"
result`,
  },
  {
    id: "exploring-data-polars",
    type: "course",
    phase: 1,
    order: 2,
    title: "Exploring Data with Polars",
    description:
      "Use Polars DataFrames for fast, expressive data exploration and analysis before modeling.",
    difficulty: "easy",
    concepts: ["polars", "DataFrame", "describe", "group_by", "filter"],
    hints: [
      "Use pl.DataFrame(data, schema=feature_names) to create a DataFrame",
      "Use .describe() to get summary statistics",
      "Use .group_by().len() for counting",
    ],
    docsUrl: [],
    code: `from sklearn.datasets import load_iris
import polars as pl

# Load iris dataset
iris = load_iris()

# TODO: Create a Polars DataFrame with proper column names
df = pl.DataFrame(None, schema=None)  # <- iris.data, iris.feature_names

# TODO: Add the target column
df = df.with_columns(None)  # <- pl.Series("species", iris.target)

print("First 5 rows:")
df.glimpse()
print()

# TODO: Get summary statistics
print("Summary statistics:")
print(None)  # <- df.describe()
print()

# TODO: Count samples per species using group_by
print("Samples per species:")
print(None)  # <- df.group_by("species").len()
print()

# Bonus: Filter to just one species
print("Only setosa (species=0):")
setosa = df.filter(pl.col("species") == 0)
print(f"  {setosa.shape[0]} samples")

result = df
result`,
    solution: `from sklearn.datasets import load_iris
import polars as pl

# Load iris dataset
iris = load_iris()

# Create a Polars DataFrame with proper column names
df = pl.DataFrame(iris.data, schema=iris.feature_names)

# Add the target column
df = df.with_columns(pl.Series("species", iris.target))

print("First 5 rows:")
df.glimpse()
print()

# Get summary statistics
print("Summary statistics:")
print(df.describe())
print()

# Count samples per species using group_by
print("Samples per species:")
print(df.group_by("species").len())
print()

# Bonus: Filter to just one species
print("Only setosa (species=0):")
setosa = df.filter(pl.col("species") == 0)
print(f"  {setosa.shape[0]} samples")

result = df
result`,
  },
  {
    id: "train-test-split",
    type: "course",
    phase: 1,
    order: 3,
    title: "Train/Test Split",
    description:
      "Learn to split data into training and test sets to evaluate model performance on unseen data.",
    difficulty: "easy",
    concepts: ["train_test_split", "test_size", "random_state", "stratify"],
    hints: [
      "train_test_split returns X_train, X_test, y_train, y_test in that order",
      "Use test_size=0.2 for 80/20 split",
      "Use random_state for reproducibility",
    ],
    docsUrl: ["https://scikit-learn.org/stable/modules/cross_validation.html"],
    code: `from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
iris = load_iris()
X, y = iris.data, iris.target

print(f"Total samples: {len(X)}")
print(f"Features per sample: {X.shape[1]}")
print()

# TODO: Split the data - 80% training, 20% testing
# Use random_state=42 for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    None, None,  # <- X and y
    test_size=None,  # <- 0.2
    random_state=None  # <- 42
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print()

# Check class distribution in splits
import numpy as np
print("Class distribution in training:", np.bincount(y_train))
print("Class distribution in test:", np.bincount(y_test))

result = f"Split: {len(X_train)} train, {len(X_test)} test"
result`,
    solution: `from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
iris = load_iris()
X, y = iris.data, iris.target

print(f"Total samples: {len(X)}")
print(f"Features per sample: {X.shape[1]}")
print()

# Split the data - 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print()

# Check class distribution in splits
import numpy as np
print("Class distribution in training:", np.bincount(y_train))
print("Class distribution in test:", np.bincount(y_test))

result = f"Split: {len(X_train)} train, {len(X_test)} test"
result`,
  },
  {
    id: "fit-predict-pattern",
    type: "course",
    phase: 1,
    order: 4,
    title: "The Fit/Predict Pattern",
    description:
      "Master sklearn's core API pattern: create a model, fit it on training data, then predict on new data.",
    difficulty: "easy",
    concepts: ["fit", "predict", "estimator API", "model training"],
    hints: [
      "All sklearn models follow: model.fit(X_train, y_train)",
      "Then use model.predict(X_test) to make predictions",
    ],
    docsUrl: ["https://scikit-learn.org/stable/getting_started.html"],
    code: `from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Prepare data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Step 1: Create the model
knn = KNeighborsClassifier(n_neighbors=3)
print("Model created:", knn)
print()

# Step 2: TODO - Fit the model on training data
None  # <- knn.fit(X_train, y_train)

# Step 3: TODO - Make predictions on test data
y_pred = None  # <- knn.predict(X_test)

# Compare predictions to actual values
print("First 10 predictions:", y_pred[:10])
print("First 10 actual:     ", y_test[:10])
print()

# Count correct predictions
correct = (y_pred == y_test).sum()
print(f"Correct predictions: {correct}/{len(y_test)}")

result = f"Predicted {len(y_pred)} samples"
result`,
    solution: `from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Prepare data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Step 1: Create the model
knn = KNeighborsClassifier(n_neighbors=3)
print("Model created:", knn)
print()

# Step 2: Fit the model on training data
knn.fit(X_train, y_train)

# Step 3: Make predictions on test data
y_pred = knn.predict(X_test)

# Compare predictions to actual values
print("First 10 predictions:", y_pred[:10])
print("First 10 actual:     ", y_test[:10])
print()

# Count correct predictions
correct = (y_pred == y_test).sum()
print(f"Correct predictions: {correct}/{len(y_test)}")

result = f"Predicted {len(y_pred)} samples"
result`,
  },
  {
    id: "knn-classifier",
    type: "course",
    phase: 1,
    order: 5,
    title: "K-Nearest Neighbors",
    description:
      "Learn K-Nearest Neighbors classification - a simple but powerful algorithm that classifies based on similarity to nearby points.",
    difficulty: "easy",
    concepts: ["KNeighborsClassifier", "n_neighbors", "distance-based"],
    hints: [
      "KNN classifies by majority vote of k nearest neighbors",
      "Try different values of n_neighbors to see the effect",
    ],
    docsUrl: ["https://scikit-learn.org/stable/modules/neighbors.html"],
    code: `from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Prepare data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# TODO: Try different values of k (n_neighbors)
k_values = [1, 3, 5, 7, 9]

print("Accuracy for different k values:")
print("-" * 30)

for k in k_values:
    # TODO: Create KNN with k neighbors, fit, and predict
    knn = None  # <- KNeighborsClassifier(n_neighbors=k)
    None  # <- fit on training data
    y_pred = None  # <- predict on test data

    # Calculate accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"k={k}: {acc:.3f}")

# Final model with best k
best_k = 5
knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X_train, y_train)

result = f"KNN with k={best_k} accuracy: {knn_final.score(X_test, y_test):.3f}"
result`,
    solution: `from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Prepare data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Try different values of k (n_neighbors)
k_values = [1, 3, 5, 7, 9]

print("Accuracy for different k values:")
print("-" * 30)

for k in k_values:
    # Create KNN with k neighbors, fit, and predict
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    # Calculate accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"k={k}: {acc:.3f}")

# Final model with best k
best_k = 5
knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X_train, y_train)

result = f"KNN with k={best_k} accuracy: {knn_final.score(X_test, y_test):.3f}"
result`,
  },
  {
    id: "decision-tree",
    type: "course",
    phase: 1,
    order: 6,
    title: "Decision Tree Classifier",
    description:
      "Learn Decision Trees - interpretable models that make predictions by learning simple decision rules from features.",
    difficulty: "easy",
    concepts: ["DecisionTreeClassifier", "max_depth", "interpretability"],
    hints: [
      "Decision trees split data based on feature thresholds",
      "Use max_depth to control tree complexity and prevent overfitting",
    ],
    docsUrl: ["https://scikit-learn.org/stable/modules/tree.html"],
    code: `from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Prepare data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# TODO: Create a Decision Tree with max_depth=3
tree = None  # <- DecisionTreeClassifier(max_depth=3, random_state=42)

# TODO: Fit and predict
None  # <- fit on training data
y_pred = None  # <- predict on test data

# Evaluate
train_acc = tree.score(X_train, y_train)
test_acc = tree.score(X_test, y_test)

print(f"Training accuracy: {train_acc:.3f}")
print(f"Test accuracy: {test_acc:.3f}")
print()

# Show tree structure
print(f"Tree depth: {tree.get_depth()}")
print(f"Number of leaves: {tree.get_n_leaves()}")
print()

# Feature importances
print("Feature importances:")
for name, importance in zip(iris.feature_names, tree.feature_importances_):
    print(f"  {name}: {importance:.3f}")

result = f"Decision Tree test accuracy: {test_acc:.3f}"
result`,
    solution: `from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Prepare data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Create a Decision Tree with max_depth=3
tree = DecisionTreeClassifier(max_depth=3, random_state=42)

# Fit and predict
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

# Evaluate
train_acc = tree.score(X_train, y_train)
test_acc = tree.score(X_test, y_test)

print(f"Training accuracy: {train_acc:.3f}")
print(f"Test accuracy: {test_acc:.3f}")
print()

# Show tree structure
print(f"Tree depth: {tree.get_depth()}")
print(f"Number of leaves: {tree.get_n_leaves()}")
print()

# Feature importances
print("Feature importances:")
for name, importance in zip(iris.feature_names, tree.feature_importances_):
    print(f"  {name}: {importance:.3f}")

result = f"Decision Tree test accuracy: {test_acc:.3f}"
result`,
  },
  {
    id: "accuracy-score",
    type: "course",
    phase: 1,
    order: 7,
    title: "Accuracy Score",
    description:
      "Learn to evaluate classifiers using accuracy - the proportion of correct predictions.",
    difficulty: "easy",
    concepts: ["accuracy_score", "score method", "evaluation metrics"],
    hints: [
      "accuracy_score(y_true, y_pred) calculates accuracy",
      "model.score(X, y) is a shortcut that returns accuracy for classifiers",
    ],
    docsUrl: ["https://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score"],
    code: `from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Load digits dataset (handwritten digits 0-9)
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42
)

print(f"Digit recognition: {len(np.unique(digits.target))} classes")
print()

# Train a classifier
clf = LogisticRegression(max_iter=5000, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# TODO: Calculate accuracy using accuracy_score function
accuracy1 = None  # <- accuracy_score(y_test, y_pred)

# TODO: Calculate accuracy using model's score method
accuracy2 = None  # <- clf.score(X_test, y_test)

print(f"Accuracy (accuracy_score): {accuracy1:.4f}")
print(f"Accuracy (model.score):    {accuracy2:.4f}")
print()

# Manual calculation for understanding
correct = (y_pred == y_test).sum()
total = len(y_test)
manual_accuracy = correct / total
print(f"Manual calculation: {correct}/{total} = {manual_accuracy:.4f}")

result = f"Test accuracy: {accuracy1:.4f}"
result`,
    solution: `from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Load digits dataset (handwritten digits 0-9)
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42
)

print(f"Digit recognition: {len(np.unique(digits.target))} classes")
print()

# Train a classifier
clf = LogisticRegression(max_iter=5000, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate accuracy using accuracy_score function
accuracy1 = accuracy_score(y_test, y_pred)

# Calculate accuracy using model's score method
accuracy2 = clf.score(X_test, y_test)

print(f"Accuracy (accuracy_score): {accuracy1:.4f}")
print(f"Accuracy (model.score):    {accuracy2:.4f}")
print()

# Manual calculation for understanding
correct = (y_pred == y_test).sum()
total = len(y_test)
manual_accuracy = correct / total
print(f"Manual calculation: {correct}/{total} = {manual_accuracy:.4f}")

result = f"Test accuracy: {accuracy1:.4f}"
result`,
  },
  {
    id: "confusion-matrix",
    type: "course",
    phase: 1,
    order: 8,
    title: "Confusion Matrix",
    description:
      "Understand confusion matrices - a detailed breakdown of correct and incorrect predictions for each class.",
    difficulty: "easy",
    concepts: ["confusion_matrix", "true positives", "false positives", "classification errors"],
    hints: [
      "Rows represent actual classes, columns represent predicted classes",
      "Diagonal elements are correct predictions",
    ],
    docsUrl: ["https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix"],
    code: `from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import numpy as np

# Prepare data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

# Train classifier
clf = DecisionTreeClassifier(max_depth=2, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# TODO: Create the confusion matrix
cm = None  # <- confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
print()

# Interpret the confusion matrix
print("Interpretation:")
print(f"  Class names: {list(iris.target_names)}")
print()
for i, class_name in enumerate(iris.target_names):
    correct = cm[i, i]
    total = cm[i, :].sum()
    print(f"  {class_name}: {correct}/{total} correct")

# Calculate per-class accuracy from confusion matrix
print()
print("Accuracy from diagonal:", cm.diagonal().sum() / cm.sum())

result = cm
result`,
    solution: `from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import numpy as np

# Prepare data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

# Train classifier
clf = DecisionTreeClassifier(max_depth=2, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
print()

# Interpret the confusion matrix
print("Interpretation:")
print(f"  Class names: {list(iris.target_names)}")
print()
for i, class_name in enumerate(iris.target_names):
    correct = cm[i, i]
    total = cm[i, :].sum()
    print(f"  {class_name}: {correct}/{total} correct")

# Calculate per-class accuracy from confusion matrix
print()
print("Accuracy from diagonal:", cm.diagonal().sum() / cm.sum())

result = cm
result`,
  },
  {
    id: "linear-regression",
    type: "course",
    phase: 1,
    order: 9,
    title: "Simple Linear Regression",
    description:
      "Learn linear regression for predicting continuous values, using the same fit/predict pattern.",
    difficulty: "easy",
    concepts: ["LinearRegression", "regression", "continuous prediction", "R2 score"],
    hints: [
      "LinearRegression uses the same fit/predict API as classifiers",
      "Use model.score() to get R-squared (coefficient of determination)",
    ],
    docsUrl: ["https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares"],
    code: `from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load diabetes dataset (regression task)
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

print(f"Features: {len(diabetes.feature_names)}")
print(f"Target: disease progression (continuous)")
print()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TODO: Create and train linear regression model
lr = None  # <- LinearRegression()
None  # <- fit on training data

# TODO: Make predictions
y_pred = None  # <- predict on test data

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.3f}")
print()

# Show some predictions vs actual
print("Sample predictions vs actual:")
for i in range(5):
    print(f"  Predicted: {y_pred[i]:.1f}, Actual: {y_test[i]:.1f}")

result = f"Linear Regression R2: {r2:.3f}"
result`,
    solution: `from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load diabetes dataset (regression task)
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

print(f"Features: {len(diabetes.feature_names)}")
print(f"Target: disease progression (continuous)")
print()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train linear regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Make predictions
y_pred = lr.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.3f}")
print()

# Show some predictions vs actual
print("Sample predictions vs actual:")
for i in range(5):
    print(f"  Predicted: {y_pred[i]:.1f}, Actual: {y_test[i]:.1f}")

result = f"Linear Regression R2: {r2:.3f}"
result`,
  },
  {
    id: "making-predictions",
    type: "course",
    phase: 1,
    order: 10,
    title: "Making Predictions on New Data",
    description:
      "Learn to use trained models to make predictions on completely new, unseen data.",
    difficulty: "easy",
    concepts: ["predict", "new data", "inference", "production use"],
    hints: [
      "New data must have the same number of features as training data",
      "Reshape single samples: X.reshape(1, -1)",
    ],
    docsUrl: ["https://scikit-learn.org/stable/getting_started.html"],
    code: `from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Train a model
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

print("Model trained on iris dataset")
print(f"Features: {iris.feature_names}")
print()

# TODO: Create a new flower measurement (completely new data)
# sepal length, sepal width, petal length, petal width
new_flower = np.array([[5.0, 3.5, 1.5, 0.2]])  # <- example measurements

# TODO: Predict the species
prediction = None  # <- clf.predict(new_flower)
predicted_species = iris.target_names[prediction[0]]

print(f"New flower measurements: {new_flower[0]}")
print(f"Predicted class: {prediction[0]}")
print(f"Predicted species: {predicted_species}")
print()

# TODO: Get prediction probabilities
probabilities = None  # <- clf.predict_proba(new_flower)

print("Prediction probabilities:")
for name, prob in zip(iris.target_names, probabilities[0]):
    print(f"  {name}: {prob:.3f}")

result = f"Predicted: {predicted_species}"
result`,
    solution: `from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Train a model
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

print("Model trained on iris dataset")
print(f"Features: {iris.feature_names}")
print()

# Create a new flower measurement (completely new data)
# sepal length, sepal width, petal length, petal width
new_flower = np.array([[5.0, 3.5, 1.5, 0.2]])

# Predict the species
prediction = clf.predict(new_flower)
predicted_species = iris.target_names[prediction[0]]

print(f"New flower measurements: {new_flower[0]}")
print(f"Predicted class: {prediction[0]}")
print(f"Predicted species: {predicted_species}")
print()

# Get prediction probabilities
probabilities = clf.predict_proba(new_flower)

print("Prediction probabilities:")
for name, prob in zip(iris.target_names, probabilities[0]):
    print(f"  {name}: {prob:.3f}")

result = f"Predicted: {predicted_species}"
result`,
  },
  {
    id: "model-persistence",
    type: "course",
    phase: 1,
    order: 11,
    title: "Understanding Model Persistence",
    description:
      "Learn how trained models can be saved and loaded for later use (conceptually - we'll use Python's pickle-like approach).",
    difficulty: "easy",
    concepts: ["model persistence", "serialization", "model state"],
    hints: [
      "Trained models store their learned parameters",
      "Models can be serialized (saved) and deserialized (loaded)",
    ],
    docsUrl: ["https://scikit-learn.org/stable/model_persistence.html"],
    code: `from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle
import io

# Train a model
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

original_model = DecisionTreeClassifier(max_depth=3, random_state=42)
original_model.fit(X_train, y_train)

original_score = original_model.score(X_test, y_test)
print(f"Original model accuracy: {original_score:.3f}")
print()

# Simulate saving and loading (in-memory for browser)
# TODO: Serialize the model to bytes
model_bytes = pickle.dumps(original_model)
print(f"Model serialized to {len(model_bytes)} bytes")

# TODO: Load the model back from bytes
loaded_model = pickle.loads(model_bytes)

# Verify the loaded model works identically
loaded_score = loaded_model.score(X_test, y_test)
print(f"Loaded model accuracy: {loaded_score:.3f}")
print()

# Verify predictions are identical
original_pred = original_model.predict(X_test[:5])
loaded_pred = loaded_model.predict(X_test[:5])

print("Original predictions:", original_pred)
print("Loaded predictions:  ", loaded_pred)
print(f"Predictions match: {all(original_pred == loaded_pred)}")

result = f"Model persistence verified: {original_score == loaded_score}"
result`,
    solution: `from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle
import io

# Train a model
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

original_model = DecisionTreeClassifier(max_depth=3, random_state=42)
original_model.fit(X_train, y_train)

original_score = original_model.score(X_test, y_test)
print(f"Original model accuracy: {original_score:.3f}")
print()

# Simulate saving and loading (in-memory for browser)
# Serialize the model to bytes
model_bytes = pickle.dumps(original_model)
print(f"Model serialized to {len(model_bytes)} bytes")

# Load the model back from bytes
loaded_model = pickle.loads(model_bytes)

# Verify the loaded model works identically
loaded_score = loaded_model.score(X_test, y_test)
print(f"Loaded model accuracy: {loaded_score:.3f}")
print()

# Verify predictions are identical
original_pred = original_model.predict(X_test[:5])
loaded_pred = loaded_model.predict(X_test[:5])

print("Original predictions:", original_pred)
print("Loaded predictions:  ", loaded_pred)
print(f"Predictions match: {all(original_pred == loaded_pred)}")

result = f"Model persistence verified: {original_score == loaded_score}"
result`,
  },
  {
    id: "multiple-features",
    type: "course",
    phase: 1,
    order: 12,
    title: "Working with Multiple Features",
    description:
      "Understand how sklearn models handle multiple features and how to inspect feature importance.",
    difficulty: "easy",
    concepts: ["features", "dimensions", "feature importance", "feature selection"],
    hints: [
      "Each column in X is a feature",
      "Some models provide feature_importances_ after fitting",
    ],
    docsUrl: ["https://scikit-learn.org/stable/modules/ensemble.html#feature-importance-evaluation"],
    code: `from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load wine dataset (13 features!)
wine = load_wine()
X, y = wine.data, wine.target

print(f"Number of features: {X.shape[1]}")
print(f"Feature names: {wine.feature_names}")
print()

# Split and train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

print(f"Model accuracy: {rf.score(X_test, y_test):.3f}")
print()

# TODO: Get feature importances
importances = None  # <- rf.feature_importances_

# TODO: Sort features by importance
sorted_idx = np.argsort(importances)[::-1]  # descending order

print("Features ranked by importance:")
for i, idx in enumerate(sorted_idx[:5], 1):
    print(f"  {i}. {wine.feature_names[idx]}: {importances[idx]:.4f}")

# Train with only top 3 features
top_3_idx = sorted_idx[:3]
X_train_top3 = X_train[:, top_3_idx]
X_test_top3 = X_test[:, top_3_idx]

rf_top3 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_top3.fit(X_train_top3, y_train)

print(f"\\nAccuracy with only top 3 features: {rf_top3.score(X_test_top3, y_test):.3f}")

result = f"Most important: {wine.feature_names[sorted_idx[0]]}"
result`,
    solution: `from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load wine dataset (13 features!)
wine = load_wine()
X, y = wine.data, wine.target

print(f"Number of features: {X.shape[1]}")
print(f"Feature names: {wine.feature_names}")
print()

# Split and train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

print(f"Model accuracy: {rf.score(X_test, y_test):.3f}")
print()

# Get feature importances
importances = rf.feature_importances_

# Sort features by importance
sorted_idx = np.argsort(importances)[::-1]  # descending order

print("Features ranked by importance:")
for i, idx in enumerate(sorted_idx[:5], 1):
    print(f"  {i}. {wine.feature_names[idx]}: {importances[idx]:.4f}")

# Train with only top 3 features
top_3_idx = sorted_idx[:3]
X_train_top3 = X_train[:, top_3_idx]
X_test_top3 = X_test[:, top_3_idx]

rf_top3 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_top3.fit(X_train_top3, y_train)

print(f"\\nAccuracy with only top 3 features: {rf_top3.score(X_test_top3, y_test):.3f}")

result = f"Most important: {wine.feature_names[sorted_idx[0]]}"
result`,
  },
  {
    id: "categorical-numerical",
    type: "course",
    phase: 1,
    order: 13,
    title: "Categorical vs Numerical Features",
    description:
      "Understand the difference between categorical and numerical features and why preprocessing matters.",
    difficulty: "easy",
    concepts: ["categorical features", "numerical features", "data types", "encoding"],
    hints: [
      "Numerical features: continuous values (age, income)",
      "Categorical features: discrete categories (color, city)",
      "Most sklearn models need numerical input",
    ],
    docsUrl: ["https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features"],
    code: `import numpy as np
import polars as pl
from sklearn.preprocessing import LabelEncoder

# Create sample data with mixed types
np.random.seed(42)
data = pl.DataFrame({
    'age': np.random.randint(20, 60, 10),           # numerical
    'income': np.random.randint(30000, 100000, 10), # numerical
    'education': np.random.choice(['high_school', 'bachelors', 'masters'], 10),  # categorical
    'city': np.random.choice(['NYC', 'LA', 'Chicago'], 10)  # categorical
})

print("Original data:")
data.glimpse()
print()

# Identify feature types
numerical_cols = ['age', 'income']
categorical_cols = ['education', 'city']

print(f"Numerical features: {numerical_cols}")
print(f"Categorical features: {categorical_cols}")
print()

# TODO: Encode categorical features using LabelEncoder
# Note: LabelEncoder expects array-like, so we use .to_numpy()
le_education = LabelEncoder()
education_encoded = None  # <- le_education.fit_transform(data["education"].to_numpy())

print("Education encoding:")
for orig, enc in zip(le_education.classes_, range(len(le_education.classes_))):
    print(f"  {orig} -> {enc}")
print()

# TODO: Encode city column
le_city = LabelEncoder()
city_encoded = None  # <- le_city.fit_transform(data["city"].to_numpy())

# Create encoded dataframe using with_columns
data_encoded = data.with_columns([
    pl.Series("education", education_encoded),
    pl.Series("city", city_encoded)
])

print("Encoded data (ready for sklearn):")
data_encoded.glimpse()

result = "Categorical features encoded successfully"
result`,
    solution: `import numpy as np
import polars as pl
from sklearn.preprocessing import LabelEncoder

# Create sample data with mixed types
np.random.seed(42)
data = pl.DataFrame({
    'age': np.random.randint(20, 60, 10),           # numerical
    'income': np.random.randint(30000, 100000, 10), # numerical
    'education': np.random.choice(['high_school', 'bachelors', 'masters'], 10),  # categorical
    'city': np.random.choice(['NYC', 'LA', 'Chicago'], 10)  # categorical
})

print("Original data:")
data.glimpse()
print()

# Identify feature types
numerical_cols = ['age', 'income']
categorical_cols = ['education', 'city']

print(f"Numerical features: {numerical_cols}")
print(f"Categorical features: {categorical_cols}")
print()

# Encode categorical features using LabelEncoder
# Note: LabelEncoder expects array-like, so we use .to_numpy()
le_education = LabelEncoder()
education_encoded = le_education.fit_transform(data["education"].to_numpy())

print("Education encoding:")
for orig, enc in zip(le_education.classes_, range(len(le_education.classes_))):
    print(f"  {orig} -> {enc}")
print()

# Encode city column
le_city = LabelEncoder()
city_encoded = le_city.fit_transform(data["city"].to_numpy())

# Create encoded dataframe using with_columns
data_encoded = data.with_columns([
    pl.Series("education", education_encoded),
    pl.Series("city", city_encoded)
])

print("Encoded data (ready for sklearn):")
data_encoded.glimpse()

result = "Categorical features encoded successfully"
result`,
  },
  {
    id: "preprocessing-intro",
    type: "course",
    phase: 1,
    order: 14,
    title: "Basic Preprocessing",
    description:
      "Introduction to data preprocessing: scaling features to improve model performance.",
    difficulty: "easy",
    concepts: ["StandardScaler", "preprocessing", "feature scaling", "normalization"],
    hints: [
      "StandardScaler transforms features to have mean=0 and std=1",
      "Always fit scaler on training data only, then transform both train and test",
    ],
    docsUrl: ["https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling"],
    code: `from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Load wine data
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42
)

# Check feature scales (they vary a lot!)
print("Feature statistics before scaling:")
print(f"  Feature 0 - Mean: {X_train[:, 0].mean():.2f}, Std: {X_train[:, 0].std():.2f}")
print(f"  Feature 1 - Mean: {X_train[:, 1].mean():.2f}, Std: {X_train[:, 1].std():.2f}")
print()

# Train KNN without scaling
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)
acc_unscaled = knn_unscaled.score(X_test, y_test)
print(f"KNN accuracy WITHOUT scaling: {acc_unscaled:.3f}")
print()

# TODO: Create and fit scaler on training data
scaler = None  # <- StandardScaler()
X_train_scaled = None  # <- scaler.fit_transform(X_train)
X_test_scaled = None  # <- scaler.transform(X_test)  # Note: only transform, not fit!

print("Feature statistics after scaling:")
print(f"  Feature 0 - Mean: {X_train_scaled[:, 0].mean():.2f}, Std: {X_train_scaled[:, 0].std():.2f}")
print(f"  Feature 1 - Mean: {X_train_scaled[:, 1].mean():.2f}, Std: {X_train_scaled[:, 1].std():.2f}")
print()

# Train KNN with scaling
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
acc_scaled = knn_scaled.score(X_test_scaled, y_test)
print(f"KNN accuracy WITH scaling: {acc_scaled:.3f}")

result = f"Improvement from scaling: {acc_scaled - acc_unscaled:.3f}"
result`,
    solution: `from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Load wine data
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42
)

# Check feature scales (they vary a lot!)
print("Feature statistics before scaling:")
print(f"  Feature 0 - Mean: {X_train[:, 0].mean():.2f}, Std: {X_train[:, 0].std():.2f}")
print(f"  Feature 1 - Mean: {X_train[:, 1].mean():.2f}, Std: {X_train[:, 1].std():.2f}")
print()

# Train KNN without scaling
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)
acc_unscaled = knn_unscaled.score(X_test, y_test)
print(f"KNN accuracy WITHOUT scaling: {acc_unscaled:.3f}")
print()

# Create and fit scaler on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Note: only transform, not fit!

print("Feature statistics after scaling:")
print(f"  Feature 0 - Mean: {X_train_scaled[:, 0].mean():.2f}, Std: {X_train_scaled[:, 0].std():.2f}")
print(f"  Feature 1 - Mean: {X_train_scaled[:, 1].mean():.2f}, Std: {X_train_scaled[:, 1].std():.2f}")
print()

# Train KNN with scaling
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
acc_scaled = knn_scaled.score(X_test_scaled, y_test)
print(f"KNN accuracy WITH scaling: {acc_scaled:.3f}")

result = f"Improvement from scaling: {acc_scaled - acc_unscaled:.3f}"
result`,
  },
  {
    id: "mini-project",
    type: "course",
    phase: 1,
    order: 15,
    title: "End-to-End Mini Project",
    description:
      "Apply everything you've learned: load data, preprocess, train, and evaluate a complete ML pipeline.",
    difficulty: "medium",
    concepts: ["complete workflow", "end-to-end ML", "best practices"],
    hints: [
      "Follow the workflow: load -> split -> preprocess -> train -> evaluate",
      "Scale your features before training distance-based models",
    ],
    docsUrl: ["https://scikit-learn.org/stable/getting_started.html"],
    code: `from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Step 1: Load data
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
print(f"Dataset: Breast Cancer Wisconsin")
print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
print(f"Classes: {cancer.target_names}")
print()

# Step 2: TODO - Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    None, None, test_size=None, random_state=42
)

# Step 3: TODO - Preprocess (scale features)
scaler = StandardScaler()
X_train_scaled = None  # <- fit_transform on training data
X_test_scaled = None  # <- transform test data

# Step 4: Train multiple models
models = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=5000, random_state=42)
}

print("Model Comparison:")
print("-" * 40)

best_model = None
best_accuracy = 0

for name, model in models.items():
    # TODO: Train each model on scaled data
    None  # <- model.fit(X_train_scaled, y_train)

    # TODO: Evaluate on test set
    accuracy = None  # <- model.score(X_test_scaled, y_test)

    print(f"{name}: {accuracy:.3f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = name

print()
print(f"Best model: {best_model} with accuracy {best_accuracy:.3f}")

result = f"Best: {best_model} ({best_accuracy:.3f})"
result`,
    solution: `from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Step 1: Load data
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
print(f"Dataset: Breast Cancer Wisconsin")
print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
print(f"Classes: {cancer.target_names}")
print()

# Step 2: Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Preprocess (scale features)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Train multiple models
models = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=5000, random_state=42)
}

print("Model Comparison:")
print("-" * 40)

best_model = None
best_accuracy = 0

for name, model in models.items():
    # Train each model on scaled data
    model.fit(X_train_scaled, y_train)

    # Evaluate on test set
    accuracy = model.score(X_test_scaled, y_test)

    print(f"{name}: {accuracy:.3f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = name

print()
print(f"Best model: {best_model} with accuracy {best_accuracy:.3f}")

result = f"Best: {best_model} ({best_accuracy:.3f})"
result`,
  },
];
