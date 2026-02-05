import type { CourseExercise } from "../types";

export const phase2Exercises: CourseExercise[] = [
  {
    id: "cross-validation",
    type: "course",
    phase: 2,
    order: 1,
    title: "Cross-Validation",
    description:
      "Learn k-fold cross-validation for more robust model evaluation than a single train/test split.",
    difficulty: "medium",
    concepts: ["cross_val_score", "k-fold", "variance estimation", "robust evaluation"],
    hints: [
      "cross_val_score automatically handles splitting and training",
      "Use cv=5 for 5-fold cross-validation",
      "Returns an array of scores, one per fold",
    ],
    docsUrl: ["https://scikit-learn.org/stable/modules/cross_validation.html"],
    code: `from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Create model
knn = KNeighborsClassifier(n_neighbors=5)

# TODO: Perform 5-fold cross-validation
scores = None  # <- cross_val_score(knn, X, y, cv=5)

print("5-Fold Cross-Validation Results:")
print(f"Individual fold scores: {scores}")
print(f"Mean accuracy: {scores.mean():.3f}")
print(f"Standard deviation: {scores.std():.3f}")
print(f"95% confidence interval: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
print()

# Compare different models
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

models = {
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(max_depth=3, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

print("Model comparison (5-fold CV):")
for name, model in models.items():
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"  {name}: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

result = f"CV mean accuracy: {scores.mean():.3f}"
result`,
    solution: `from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Create model
knn = KNeighborsClassifier(n_neighbors=5)

# Perform 5-fold cross-validation
scores = cross_val_score(knn, X, y, cv=5)

print("5-Fold Cross-Validation Results:")
print(f"Individual fold scores: {scores}")
print(f"Mean accuracy: {scores.mean():.3f}")
print(f"Standard deviation: {scores.std():.3f}")
print(f"95% confidence interval: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
print()

# Compare different models
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

models = {
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(max_depth=3, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

print("Model comparison (5-fold CV):")
for name, model in models.items():
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"  {name}: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

result = f"CV mean accuracy: {scores.mean():.3f}"
result`,
  },
  {
    id: "standard-scaler",
    type: "course",
    phase: 2,
    order: 2,
    title: "StandardScaler Deep Dive",
    description:
      "Master StandardScaler: understand how it works and when to use it.",
    difficulty: "medium",
    concepts: ["StandardScaler", "z-score", "fit vs transform", "data leakage"],
    hints: [
      "StandardScaler computes: (x - mean) / std",
      "Always fit on training data only to avoid data leakage",
      "The scaler stores mean_ and scale_ attributes after fitting",
    ],
    docsUrl: ["https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling"],
    code: `from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load data
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42
)

# Create and fit scaler
scaler = StandardScaler()

# TODO: Fit scaler on training data only
None  # <- scaler.fit(X_train)

print("Scaler learned parameters (first 3 features):")
print(f"  Means: {scaler.mean_[:3]}")
print(f"  Scales (std): {scaler.scale_[:3]}")
print()

# TODO: Transform both train and test data
X_train_scaled = None  # <- scaler.transform(X_train)
X_test_scaled = None  # <- scaler.transform(X_test)

# Verify the transformation
print("Training data after scaling (first 3 features):")
print(f"  Mean: {X_train_scaled[:, :3].mean(axis=0)}")
print(f"  Std: {X_train_scaled[:, :3].std(axis=0)}")
print()

# Note: test data won't have exactly 0 mean / 1 std (that's expected!)
print("Test data after scaling (first 3 features):")
print(f"  Mean: {X_test_scaled[:, :3].mean(axis=0)}")
print(f"  Std: {X_test_scaled[:, :3].std(axis=0)}")
print()

# Manual verification
feature_0 = X_train[:, 0]
manual_scaled = (feature_0 - scaler.mean_[0]) / scaler.scale_[0]
print("Manual scaling verification (feature 0):")
print(f"  Scaler result: {X_train_scaled[0, 0]:.6f}")
print(f"  Manual result: {manual_scaled[0]:.6f}")

result = f"Scaler fitted with mean={scaler.mean_[0]:.2f}, scale={scaler.scale_[0]:.2f}"
result`,
    solution: `from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load data
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42
)

# Create and fit scaler
scaler = StandardScaler()

# Fit scaler on training data only
scaler.fit(X_train)

print("Scaler learned parameters (first 3 features):")
print(f"  Means: {scaler.mean_[:3]}")
print(f"  Scales (std): {scaler.scale_[:3]}")
print()

# Transform both train and test data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Verify the transformation
print("Training data after scaling (first 3 features):")
print(f"  Mean: {X_train_scaled[:, :3].mean(axis=0)}")
print(f"  Std: {X_train_scaled[:, :3].std(axis=0)}")
print()

# Note: test data won't have exactly 0 mean / 1 std (that's expected!)
print("Test data after scaling (first 3 features):")
print(f"  Mean: {X_test_scaled[:, :3].mean(axis=0)}")
print(f"  Std: {X_test_scaled[:, :3].std(axis=0)}")
print()

# Manual verification
feature_0 = X_train[:, 0]
manual_scaled = (feature_0 - scaler.mean_[0]) / scaler.scale_[0]
print("Manual scaling verification (feature 0):")
print(f"  Scaler result: {X_train_scaled[0, 0]:.6f}")
print(f"  Manual result: {manual_scaled[0]:.6f}")

result = f"Scaler fitted with mean={scaler.mean_[0]:.2f}, scale={scaler.scale_[0]:.2f}"
result`,
  },
  {
    id: "one-hot-encoder",
    type: "course",
    phase: 2,
    order: 3,
    title: "OneHotEncoder for Categories",
    description:
      "Learn OneHotEncoder to convert categorical features into numerical format for ML models.",
    difficulty: "medium",
    concepts: ["OneHotEncoder", "categorical encoding", "sparse matrix", "drop parameter"],
    hints: [
      "OneHotEncoder creates binary columns for each category",
      "Use sparse_output=False to get a dense numpy array",
      "handle_unknown='ignore' handles unseen categories during transform",
    ],
    docsUrl: ["https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features"],
    code: `from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Sample categorical data
categories = np.array([['red', 'small'],
                       ['blue', 'medium'],
                       ['green', 'large'],
                       ['red', 'medium'],
                       ['blue', 'small']])

print("Original data:")
print(categories)
print()

# TODO: Create OneHotEncoder with sparse_output=False
encoder = None  # <- OneHotEncoder(sparse_output=False)

# TODO: Fit and transform
encoded = None  # <- encoder.fit_transform(categories)

print("Encoded data:")
print(encoded)
print()

# Understand the encoding
print("Categories per feature:")
for i, cats in enumerate(encoder.categories_):
    print(f"  Feature {i}: {cats}")
print()

print("Feature names:")
print(encoder.get_feature_names_out(['color', 'size']))
print()

# Handle unknown categories
new_data = np.array([['red', 'small'], ['yellow', 'xlarge']])
encoder_safe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder_safe.fit(categories)

print("Encoding new data with unknown categories:")
print(f"  Input: {new_data}")
print(f"  Output: {encoder_safe.transform(new_data)}")

result = f"Encoded shape: {encoded.shape}"
result`,
    solution: `from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Sample categorical data
categories = np.array([['red', 'small'],
                       ['blue', 'medium'],
                       ['green', 'large'],
                       ['red', 'medium'],
                       ['blue', 'small']])

print("Original data:")
print(categories)
print()

# Create OneHotEncoder with sparse_output=False
encoder = OneHotEncoder(sparse_output=False)

# Fit and transform
encoded = encoder.fit_transform(categories)

print("Encoded data:")
print(encoded)
print()

# Understand the encoding
print("Categories per feature:")
for i, cats in enumerate(encoder.categories_):
    print(f"  Feature {i}: {cats}")
print()

print("Feature names:")
print(encoder.get_feature_names_out(['color', 'size']))
print()

# Handle unknown categories
new_data = np.array([['red', 'small'], ['yellow', 'xlarge']])
encoder_safe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder_safe.fit(categories)

print("Encoding new data with unknown categories:")
print(f"  Input: {new_data}")
print(f"  Output: {encoder_safe.transform(new_data)}")

result = f"Encoded shape: {encoded.shape}"
result`,
  },
  {
    id: "pipelines",
    type: "course",
    phase: 2,
    order: 4,
    title: "Building Pipelines",
    description:
      "Learn to create scikit-learn Pipelines that chain preprocessing and modeling steps.",
    difficulty: "medium",
    concepts: ["Pipeline", "make_pipeline", "chaining transformers", "fit_transform"],
    hints: [
      "Pipeline ensures correct order: fit transforms, then fit model",
      "Prevents data leakage by handling train/test properly",
      "Access steps with pipe.named_steps or pipe['step_name']",
    ],
    docsUrl: ["https://scikit-learn.org/stable/modules/compose.html#pipeline"],
    code: `from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

# Load data
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42
)

# TODO: Create a pipeline with StandardScaler and LogisticRegression
pipe = Pipeline([
    ('scaler', None),  # <- StandardScaler()
    ('classifier', None)  # <- LogisticRegression(max_iter=1000, random_state=42)
])

# TODO: Fit the entire pipeline
None  # <- pipe.fit(X_train, y_train)

# Evaluate
print(f"Pipeline test accuracy: {pipe.score(X_test, y_test):.3f}")
print()

# Cross-validation works seamlessly with pipelines
cv_scores = cross_val_score(pipe, wine.data, wine.target, cv=5)
print(f"Pipeline CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
print()

# Access pipeline steps
print("Pipeline steps:")
for name, step in pipe.named_steps.items():
    print(f"  {name}: {step.__class__.__name__}")
print()

# Shorter syntax with make_pipeline
pipe2 = make_pipeline(
    StandardScaler(),
    PCA(n_components=5),
    LogisticRegression(max_iter=1000, random_state=42)
)
pipe2.fit(X_train, y_train)
print(f"Pipeline with PCA: {pipe2.score(X_test, y_test):.3f}")

result = f"Pipeline accuracy: {pipe.score(X_test, y_test):.3f}"
result`,
    solution: `from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

# Load data
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42
)

# Create a pipeline with StandardScaler and LogisticRegression
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# Fit the entire pipeline
pipe.fit(X_train, y_train)

# Evaluate
print(f"Pipeline test accuracy: {pipe.score(X_test, y_test):.3f}")
print()

# Cross-validation works seamlessly with pipelines
cv_scores = cross_val_score(pipe, wine.data, wine.target, cv=5)
print(f"Pipeline CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
print()

# Access pipeline steps
print("Pipeline steps:")
for name, step in pipe.named_steps.items():
    print(f"  {name}: {step.__class__.__name__}")
print()

# Shorter syntax with make_pipeline
pipe2 = make_pipeline(
    StandardScaler(),
    PCA(n_components=5),
    LogisticRegression(max_iter=1000, random_state=42)
)
pipe2.fit(X_train, y_train)
print(f"Pipeline with PCA: {pipe2.score(X_test, y_test):.3f}")

result = f"Pipeline accuracy: {pipe.score(X_test, y_test):.3f}"
result`,
  },
  {
    id: "column-transformer",
    type: "course",
    phase: 2,
    order: 5,
    title: "ColumnTransformer for Mixed Data",
    description:
      "Use ColumnTransformer to apply different preprocessing to different columns.",
    difficulty: "medium",
    concepts: ["ColumnTransformer", "mixed data types", "parallel preprocessing"],
    hints: [
      "ColumnTransformer applies different transformers to different columns",
      "Use 'passthrough' to keep columns unchanged",
      "remainder='passthrough' or 'drop' handles unspecified columns",
      "Convert polars to pandas for ColumnTransformer (uses column names)",
    ],
    docsUrl: ["https://scikit-learn.org/stable/modules/compose.html#columntransformer-for-heterogeneous-data"],
    code: `from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
import numpy as np
import polars as pl

# Create mixed dataset with Polars
np.random.seed(42)
df = pl.DataFrame({
    'age': np.random.randint(18, 65, 100),
    'salary': np.random.randint(30000, 120000, 100),
    'department': np.random.choice(['Sales', 'Engineering', 'HR'], 100),
    'level': np.random.choice(['Junior', 'Senior'], 100),
    'promoted': np.random.randint(0, 2, 100)
})

print("Data preview (Polars):")
df.glimpse()
print()

# ColumnTransformer works best with pandas (uses column names)
# Convert for sklearn preprocessing
X = df.drop('promoted').to_pandas()
y = df['promoted'].to_numpy()

# Define column types
numeric_features = ['age', 'salary']
categorical_features = ['department', 'level']

# TODO: Create ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', None, numeric_features),  # <- StandardScaler()
        ('cat', None, categorical_features)  # <- OneHotEncoder(handle_unknown='ignore')
    ]
)

# TODO: Create full pipeline with preprocessor and classifier
pipe = Pipeline([
    ('preprocessor', None),  # <- preprocessor
    ('classifier', LogisticRegression(random_state=42))
])

# Fit and evaluate
pipe.fit(X, y)
print(f"Training accuracy: {pipe.score(X, y):.3f}")
print()

# Inspect transformed features
X_transformed = preprocessor.fit_transform(X)
print(f"Original features: {X.shape[1]}")
print(f"Transformed features: {X_transformed.shape[1]}")
print()

# Get feature names
cat_encoder = preprocessor.named_transformers_['cat']
cat_features = cat_encoder.get_feature_names_out(categorical_features)
all_features = numeric_features + list(cat_features)
print(f"All feature names: {all_features}")

result = f"Preprocessed {X.shape[1]} -> {X_transformed.shape[1]} features"
result`,
    solution: `from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
import numpy as np
import polars as pl

# Create mixed dataset with Polars
np.random.seed(42)
df = pl.DataFrame({
    'age': np.random.randint(18, 65, 100),
    'salary': np.random.randint(30000, 120000, 100),
    'department': np.random.choice(['Sales', 'Engineering', 'HR'], 100),
    'level': np.random.choice(['Junior', 'Senior'], 100),
    'promoted': np.random.randint(0, 2, 100)
})

print("Data preview (Polars):")
df.glimpse()
print()

# ColumnTransformer works best with pandas (uses column names)
# Convert for sklearn preprocessing
X = df.drop('promoted').to_pandas()
y = df['promoted'].to_numpy()

# Define column types
numeric_features = ['age', 'salary']
categorical_features = ['department', 'level']

# Create ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Create full pipeline with preprocessor and classifier
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))
])

# Fit and evaluate
pipe.fit(X, y)
print(f"Training accuracy: {pipe.score(X, y):.3f}")
print()

# Inspect transformed features
X_transformed = preprocessor.fit_transform(X)
print(f"Original features: {X.shape[1]}")
print(f"Transformed features: {X_transformed.shape[1]}")
print()

# Get feature names
cat_encoder = preprocessor.named_transformers_['cat']
cat_features = cat_encoder.get_feature_names_out(categorical_features)
all_features = numeric_features + list(cat_features)
print(f"All feature names: {all_features}")

result = f"Preprocessed {X.shape[1]} -> {X_transformed.shape[1]} features"
result`,
  },
  {
    id: "precision-recall-f1",
    type: "course",
    phase: 2,
    order: 6,
    title: "Precision, Recall, and F1",
    description:
      "Learn precision, recall, and F1-score for better evaluation of classification models.",
    difficulty: "medium",
    concepts: ["precision", "recall", "F1-score", "classification_report"],
    hints: [
      "Precision: of all predicted positive, how many are correct",
      "Recall: of all actual positive, how many did we find",
      "F1: harmonic mean of precision and recall",
    ],
    docsUrl: ["https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-and-f-measures"],
    code: `from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import numpy as np

# Load data
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.3, random_state=42
)

# Train model
clf = LogisticRegression(max_iter=5000, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# TODO: Calculate precision, recall, and F1
precision = None  # <- precision_score(y_test, y_pred)
recall = None  # <- recall_score(y_test, y_pred)
f1 = None  # <- f1_score(y_test, y_pred)

print("Individual Metrics:")
print(f"  Precision: {precision:.3f}")
print(f"  Recall: {recall:.3f}")
print(f"  F1-Score: {f1:.3f}")
print()

# Understanding the metrics
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("From Confusion Matrix:")
print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
print(f"  Precision = TP/(TP+FP) = {tp}/{tp+fp} = {tp/(tp+fp):.3f}")
print(f"  Recall = TP/(TP+FN) = {tp}/{tp+fn} = {tp/(tp+fn):.3f}")
print()

# Full classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))

result = f"F1-Score: {f1:.3f}"
result`,
    solution: `from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import numpy as np

# Load data
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.3, random_state=42
)

# Train model
clf = LogisticRegression(max_iter=5000, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Calculate precision, recall, and F1
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Individual Metrics:")
print(f"  Precision: {precision:.3f}")
print(f"  Recall: {recall:.3f}")
print(f"  F1-Score: {f1:.3f}")
print()

# Understanding the metrics
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("From Confusion Matrix:")
print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
print(f"  Precision = TP/(TP+FP) = {tp}/{tp+fp} = {tp/(tp+fp):.3f}")
print(f"  Recall = TP/(TP+FN) = {tp}/{tp+fn} = {tp/(tp+fn):.3f}")
print()

# Full classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))

result = f"F1-Score: {f1:.3f}"
result`,
  },
  {
    id: "roc-auc",
    type: "course",
    phase: 2,
    order: 7,
    title: "ROC Curves and AUC",
    description:
      "Learn ROC curves and AUC for threshold-independent model evaluation.",
    difficulty: "medium",
    concepts: ["ROC curve", "AUC", "threshold", "predict_proba"],
    hints: [
      "ROC plots True Positive Rate vs False Positive Rate",
      "AUC = 1.0 is perfect, AUC = 0.5 is random",
      "Use predict_proba to get probability scores",
    ],
    docsUrl: ["https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics"],
    code: `from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

# Load and split data
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.3, random_state=42
)

# Train model
clf = LogisticRegression(max_iter=5000, random_state=42)
clf.fit(X_train, y_train)

# TODO: Get probability predictions for positive class
y_prob = None  # <- clf.predict_proba(X_test)[:, 1]

# TODO: Calculate ROC curve points
fpr, tpr, thresholds = None  # <- roc_curve(y_test, y_prob)

# TODO: Calculate AUC
roc_auc = None  # <- auc(fpr, tpr) or roc_auc_score(y_test, y_prob)

print(f"ROC AUC Score: {roc_auc:.3f}")
print()

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Threshold examples:")
for i in range(0, len(thresholds), len(thresholds)//5):
    print(f"  Threshold={thresholds[i]:.3f}: FPR={fpr[i]:.3f}, TPR={tpr[i]:.3f}")

result = f"AUC: {roc_auc:.3f}"
result`,
    solution: `from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

# Load and split data
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.3, random_state=42
)

# Train model
clf = LogisticRegression(max_iter=5000, random_state=42)
clf.fit(X_train, y_train)

# Get probability predictions for positive class
y_prob = clf.predict_proba(X_test)[:, 1]

# Calculate ROC curve points
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Calculate AUC
roc_auc = auc(fpr, tpr)

print(f"ROC AUC Score: {roc_auc:.3f}")
print()

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Threshold examples:")
for i in range(0, len(thresholds), len(thresholds)//5):
    print(f"  Threshold={thresholds[i]:.3f}: FPR={fpr[i]:.3f}, TPR={tpr[i]:.3f}")

result = f"AUC: {roc_auc:.3f}"
result`,
  },
  {
    id: "ridge-lasso",
    type: "course",
    phase: 2,
    order: 8,
    title: "Ridge & Lasso Regularization",
    description:
      "Learn regularization techniques to prevent overfitting in linear models.",
    difficulty: "medium",
    concepts: ["Ridge", "Lasso", "regularization", "alpha", "overfitting"],
    hints: [
      "Ridge (L2): shrinks coefficients but keeps all features",
      "Lasso (L1): can reduce some coefficients to exactly zero (feature selection)",
      "Higher alpha = stronger regularization",
    ],
    docsUrl: [
      "https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression",
      "https://scikit-learn.org/stable/modules/linear_model.html#lasso"
    ],
    code: `from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load data
diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, test_size=0.2, random_state=42
)

# Standard Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

print("Linear Regression:")
print(f"  R2: {r2_score(y_test, lr_pred):.3f}")
print(f"  Max coefficient: {np.abs(lr.coef_).max():.2f}")
print()

# TODO: Ridge Regression (L2 regularization)
ridge = None  # <- Ridge(alpha=1.0)
None  # <- fit on training data
ridge_pred = ridge.predict(X_test)

print("Ridge Regression (alpha=1.0):")
print(f"  R2: {r2_score(y_test, ridge_pred):.3f}")
print(f"  Max coefficient: {np.abs(ridge.coef_).max():.2f}")
print()

# TODO: Lasso Regression (L1 regularization)
lasso = None  # <- Lasso(alpha=1.0)
None  # <- fit on training data
lasso_pred = lasso.predict(X_test)

print("Lasso Regression (alpha=1.0):")
print(f"  R2: {r2_score(y_test, lasso_pred):.3f}")
print(f"  Max coefficient: {np.abs(lasso.coef_).max():.2f}")
print(f"  Zero coefficients: {(lasso.coef_ == 0).sum()}/{len(lasso.coef_)}")
print()

# Compare coefficients
print("Coefficient comparison (first 5 features):")
print(f"  Linear: {lr.coef_[:5].round(2)}")
print(f"  Ridge:  {ridge.coef_[:5].round(2)}")
print(f"  Lasso:  {lasso.coef_[:5].round(2)}")

result = f"Lasso zeroed {(lasso.coef_ == 0).sum()} features"
result`,
    solution: `from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load data
diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, test_size=0.2, random_state=42
)

# Standard Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

print("Linear Regression:")
print(f"  R2: {r2_score(y_test, lr_pred):.3f}")
print(f"  Max coefficient: {np.abs(lr.coef_).max():.2f}")
print()

# Ridge Regression (L2 regularization)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)

print("Ridge Regression (alpha=1.0):")
print(f"  R2: {r2_score(y_test, ridge_pred):.3f}")
print(f"  Max coefficient: {np.abs(ridge.coef_).max():.2f}")
print()

# Lasso Regression (L1 regularization)
lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)

print("Lasso Regression (alpha=1.0):")
print(f"  R2: {r2_score(y_test, lasso_pred):.3f}")
print(f"  Max coefficient: {np.abs(lasso.coef_).max():.2f}")
print(f"  Zero coefficients: {(lasso.coef_ == 0).sum()}/{len(lasso.coef_)}")
print()

# Compare coefficients
print("Coefficient comparison (first 5 features):")
print(f"  Linear: {lr.coef_[:5].round(2)}")
print(f"  Ridge:  {ridge.coef_[:5].round(2)}")
print(f"  Lasso:  {lasso.coef_[:5].round(2)}")

result = f"Lasso zeroed {(lasso.coef_ == 0).sum()} features"
result`,
  },
  {
    id: "grid-search-cv",
    type: "course",
    phase: 2,
    order: 9,
    title: "GridSearchCV Basics",
    description:
      "Learn to systematically search for optimal hyperparameters using GridSearchCV.",
    difficulty: "medium",
    concepts: ["GridSearchCV", "hyperparameter tuning", "parameter grid", "best_params_"],
    hints: [
      "Define a parameter grid as a dictionary",
      "GridSearchCV tries all combinations and uses cross-validation",
      "Access results with best_params_, best_score_, cv_results_",
    ],
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
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

print(f"Total combinations: {3 * 2 * 2} = 12")
print()

# TODO: Create GridSearchCV
grid_search = GridSearchCV(
    None,  # <- SVC()
    param_grid,
    cv=None,  # <- 5
    scoring='accuracy',
    return_train_score=True
)

# TODO: Fit grid search
None  # <- grid_search.fit(X_train, y_train)

print("Grid Search Results:")
print(f"  Best parameters: {grid_search.best_params_}")
print(f"  Best CV score: {grid_search.best_score_:.3f}")
print(f"  Test score: {grid_search.score(X_test, y_test):.3f}")
print()

# Show top 5 parameter combinations
import pandas as pd
results = pd.DataFrame(grid_search.cv_results_)
results = results.sort_values('rank_test_score')
print("Top 5 parameter combinations:")
for _, row in results.head().iterrows():
    print(f"  {row['params']}: {row['mean_test_score']:.3f}")

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
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

print(f"Total combinations: {3 * 2 * 2} = 12")
print()

# Create GridSearchCV
grid_search = GridSearchCV(
    SVC(),
    param_grid,
    cv=5,
    scoring='accuracy',
    return_train_score=True
)

# Fit grid search
grid_search.fit(X_train, y_train)

print("Grid Search Results:")
print(f"  Best parameters: {grid_search.best_params_}")
print(f"  Best CV score: {grid_search.best_score_:.3f}")
print(f"  Test score: {grid_search.score(X_test, y_test):.3f}")
print()

# Show top 5 parameter combinations
import pandas as pd
results = pd.DataFrame(grid_search.cv_results_)
results = results.sort_values('rank_test_score')
print("Top 5 parameter combinations:")
for _, row in results.head().iterrows():
    print(f"  {row['params']}: {row['mean_test_score']:.3f}")

result = f"Best params: {grid_search.best_params_}"
result`,
  },
  {
    id: "randomized-search",
    type: "course",
    phase: 2,
    order: 10,
    title: "RandomizedSearchCV",
    description:
      "Learn RandomizedSearchCV for efficient hyperparameter search with many parameters.",
    difficulty: "medium",
    concepts: ["RandomizedSearchCV", "distributions", "n_iter", "efficiency"],
    hints: [
      "RandomizedSearchCV samples from parameter distributions",
      "More efficient than GridSearchCV for large parameter spaces",
      "Use n_iter to control how many combinations to try",
    ],
    docsUrl: ["https://scikit-learn.org/stable/modules/grid_search.html#randomized-parameter-optimization"],
    code: `from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint, uniform
import numpy as np

# Load data
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42
)

# Define parameter distributions
param_distributions = {
    'n_estimators': randint(10, 200),           # uniform integer [10, 200)
    'max_depth': randint(3, 20),                # uniform integer [3, 20)
    'min_samples_split': randint(2, 20),        # uniform integer [2, 20)
    'min_samples_leaf': randint(1, 10),         # uniform integer [1, 10)
}

print("Parameter distributions defined")
print(f"  n_estimators: [10, 200)")
print(f"  max_depth: [3, 20)")
print()

# TODO: Create RandomizedSearchCV
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions,
    n_iter=None,  # <- 20 (try 20 random combinations)
    cv=None,  # <- 3
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

# TODO: Fit random search
None  # <- random_search.fit(X_train, y_train)

print("Randomized Search Results:")
print(f"  Best parameters: {random_search.best_params_}")
print(f"  Best CV score: {random_search.best_score_:.3f}")
print(f"  Test score: {random_search.score(X_test, y_test):.3f}")
print()

# Compare to default model
default_rf = RandomForestClassifier(random_state=42)
default_rf.fit(X_train, y_train)
print(f"Default RF test score: {default_rf.score(X_test, y_test):.3f}")
print(f"Tuned RF test score: {random_search.score(X_test, y_test):.3f}")

result = f"Improvement: {random_search.score(X_test, y_test) - default_rf.score(X_test, y_test):.3f}"
result`,
    solution: `from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint, uniform
import numpy as np

# Load data
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42
)

# Define parameter distributions
param_distributions = {
    'n_estimators': randint(10, 200),           # uniform integer [10, 200)
    'max_depth': randint(3, 20),                # uniform integer [3, 20)
    'min_samples_split': randint(2, 20),        # uniform integer [2, 20)
    'min_samples_leaf': randint(1, 10),         # uniform integer [1, 10)
}

print("Parameter distributions defined")
print(f"  n_estimators: [10, 200)")
print(f"  max_depth: [3, 20)")
print()

# Create RandomizedSearchCV
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions,
    n_iter=20,  # try 20 random combinations
    cv=3,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

# Fit random search
random_search.fit(X_train, y_train)

print("Randomized Search Results:")
print(f"  Best parameters: {random_search.best_params_}")
print(f"  Best CV score: {random_search.best_score_:.3f}")
print(f"  Test score: {random_search.score(X_test, y_test):.3f}")
print()

# Compare to default model
default_rf = RandomForestClassifier(random_state=42)
default_rf.fit(X_train, y_train)
print(f"Default RF test score: {default_rf.score(X_test, y_test):.3f}")
print(f"Tuned RF test score: {random_search.score(X_test, y_test):.3f}")

result = f"Improvement: {random_search.score(X_test, y_test) - default_rf.score(X_test, y_test):.3f}"
result`,
  },
  {
    id: "random-forest-intro",
    type: "course",
    phase: 2,
    order: 11,
    title: "Random Forest Introduction",
    description:
      "Learn Random Forest - a powerful ensemble method combining many decision trees.",
    difficulty: "medium",
    concepts: ["RandomForestClassifier", "ensemble", "bagging", "feature importance"],
    hints: [
      "Random Forest trains many trees on random subsets of data",
      "Final prediction is majority vote (classification) or average (regression)",
      "n_estimators controls the number of trees",
    ],
    docsUrl: ["https://scikit-learn.org/stable/modules/ensemble.html#random-forests"],
    code: `from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Load data
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42
)

# Single Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_score = dt.score(X_test, y_test)

# TODO: Create Random Forest with 100 trees
rf = None  # <- RandomForestClassifier(n_estimators=100, random_state=42)

# TODO: Fit and score
None  # <- rf.fit(X_train, y_train)
rf_score = None  # <- rf.score(X_test, y_test)

print("Single Tree vs Random Forest:")
print(f"  Decision Tree: {dt_score:.3f}")
print(f"  Random Forest: {rf_score:.3f}")
print()

# Feature importance from Random Forest
importances = rf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

print("Top 5 important features:")
for i in range(5):
    idx = sorted_idx[i]
    print(f"  {wine.feature_names[idx]}: {importances[idx]:.4f}")
print()

# Cross-validation comparison
dt_cv = cross_val_score(dt, wine.data, wine.target, cv=5)
rf_cv = cross_val_score(rf, wine.data, wine.target, cv=5)

print("Cross-validation scores:")
print(f"  Decision Tree: {dt_cv.mean():.3f} (+/- {dt_cv.std() * 2:.3f})")
print(f"  Random Forest: {rf_cv.mean():.3f} (+/- {rf_cv.std() * 2:.3f})")

result = f"RF accuracy: {rf_score:.3f}"
result`,
    solution: `from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Load data
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42
)

# Single Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_score = dt.score(X_test, y_test)

# Create Random Forest with 100 trees
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit and score
rf.fit(X_train, y_train)
rf_score = rf.score(X_test, y_test)

print("Single Tree vs Random Forest:")
print(f"  Decision Tree: {dt_score:.3f}")
print(f"  Random Forest: {rf_score:.3f}")
print()

# Feature importance from Random Forest
importances = rf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

print("Top 5 important features:")
for i in range(5):
    idx = sorted_idx[i]
    print(f"  {wine.feature_names[idx]}: {importances[idx]:.4f}")
print()

# Cross-validation comparison
dt_cv = cross_val_score(dt, wine.data, wine.target, cv=5)
rf_cv = cross_val_score(rf, wine.data, wine.target, cv=5)

print("Cross-validation scores:")
print(f"  Decision Tree: {dt_cv.mean():.3f} (+/- {dt_cv.std() * 2:.3f})")
print(f"  Random Forest: {rf_cv.mean():.3f} (+/- {rf_cv.std() * 2:.3f})")

result = f"RF accuracy: {rf_score:.3f}"
result`,
  },
  {
    id: "feature-importance",
    type: "course",
    phase: 2,
    order: 12,
    title: "Feature Importance Analysis",
    description:
      "Learn to extract and interpret feature importances from tree-based models.",
    difficulty: "medium",
    concepts: ["feature_importances_", "feature ranking", "model interpretation"],
    hints: [
      "Tree-based models provide feature_importances_ after fitting",
      "Importances sum to 1.0",
      "Use for feature selection and model interpretation",
    ],
    docsUrl: ["https://scikit-learn.org/stable/modules/ensemble.html#feature-importance-evaluation"],
    code: `from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Load data
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42
)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# TODO: Get feature importances
importances = None  # <- rf.feature_importances_

# TODO: Get indices that would sort importances (descending)
sorted_idx = None  # <- np.argsort(importances)[::-1]

print("Top 10 most important features:")
print("-" * 50)
for i in range(10):
    idx = sorted_idx[i]
    print(f"{i+1:2}. {cancer.feature_names[idx]:25} {importances[idx]:.4f}")
print()

# Visualize top 10 features
top_10_idx = sorted_idx[:10]
top_10_importances = importances[top_10_idx]
top_10_names = [cancer.feature_names[i] for i in top_10_idx]

plt.figure(figsize=(10, 6))
plt.barh(range(10), top_10_importances[::-1])
plt.yticks(range(10), top_10_names[::-1])
plt.xlabel('Feature Importance')
plt.title('Top 10 Feature Importances (Random Forest)')
plt.tight_layout()
plt.show()

# Cumulative importance
cumsum = np.cumsum(importances[sorted_idx])
n_features_90 = np.argmax(cumsum >= 0.9) + 1
print(f"\\nFeatures needed for 90% importance: {n_features_90}")
print(f"Total features: {len(importances)}")

result = f"Top feature: {cancer.feature_names[sorted_idx[0]]}"
result`,
    solution: `from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Load data
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42
)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get feature importances
importances = rf.feature_importances_

# Get indices that would sort importances (descending)
sorted_idx = np.argsort(importances)[::-1]

print("Top 10 most important features:")
print("-" * 50)
for i in range(10):
    idx = sorted_idx[i]
    print(f"{i+1:2}. {cancer.feature_names[idx]:25} {importances[idx]:.4f}")
print()

# Visualize top 10 features
top_10_idx = sorted_idx[:10]
top_10_importances = importances[top_10_idx]
top_10_names = [cancer.feature_names[i] for i in top_10_idx]

plt.figure(figsize=(10, 6))
plt.barh(range(10), top_10_importances[::-1])
plt.yticks(range(10), top_10_names[::-1])
plt.xlabel('Feature Importance')
plt.title('Top 10 Feature Importances (Random Forest)')
plt.tight_layout()
plt.show()

# Cumulative importance
cumsum = np.cumsum(importances[sorted_idx])
n_features_90 = np.argmax(cumsum >= 0.9) + 1
print(f"\\nFeatures needed for 90% importance: {n_features_90}")
print(f"Total features: {len(importances)}")

result = f"Top feature: {cancer.feature_names[sorted_idx[0]]}"
result`,
  },
];
