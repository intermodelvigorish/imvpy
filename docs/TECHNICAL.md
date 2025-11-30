# IMV Package - Technical Documentation

## Table of Contents

1. [Overview](#overview)
2. [Package Architecture](#package-architecture)
3. [Module: imv/core.py](#module-imvcorepy)
4. [Module: imv/binary.py](#module-imvbinarypy)
5. [Module: imv/multiclass.py](#module-imvmulticlasspy)
6. [Module: imv/ablation.py](#module-imvablationpy)
7. [Mathematical Foundation](#mathematical-foundation)
8. [Implementation Details](#implementation-details)
9. [Usage Examples](#usage-examples)
10. [Performance Considerations](#performance-considerations)

---

## Overview

The **IMV (Information Model Vigor)** package provides a unified framework for quantifying the information contribution of features in machine learning models across different contexts:

- **Binary Classification**: SHAP-IMV with Shapley value attribution
- **Multiclass Classification**: Pairwise IMV matrices for multi-outcome problems
- **Deep Learning**: Ablation studies for neural network architectures

### Key Innovations

1. **Unified Core Functions**: All IMV variants share the same mathematical foundation in `imv/core.py`
2. **DRY Principle**: Eliminates code duplication (~250 lines) from previous structure
3. **Backward Compatible**: Old class names preserved as aliases
4. **Flexible API**: Works with any scikit-learn compatible model

---

## Package Architecture

### New Structure (Post-Reorganization)

```
imv/
├── __init__.py          # Package exports and API
├── core.py              # Shared IMV mathematical functions
├── binary.py            # Binary classification IMV (SHAP)
├── multiclass.py        # Multiclass classification IMV
├── ablation.py          # Deep learning ablation studies
├── imv.py              # Original IMV utilities
└── utils.py            # Helper functions
```

### Design Principles

1. **Single Source of Truth**: Core IMV functions (`ll()`, `get_w()`, etc.) defined once
2. **Separation of Concerns**: Each module handles a specific use case
3. **Composability**: Modules import from core rather than duplicating code
4. **Testability**: Core functions can be tested independently

### Import Patterns

```python
# Modern imports
from imv import BinaryIMV, MulticlassIMV, AblationIMV
from imv import ll, get_w, calculate_imv

# Backward compatible
from imv import IMVEvaluator, MultinomialIMV
```

---

## Module: imv/core.py

### Purpose

Contains the fundamental mathematical functions for IMV calculation. All other modules import from here to avoid duplication.

### Key Functions

#### `ll(x, p, epsilon=1e-9)`

**Log-Likelihood Geometric Mean**

Calculates the geometric mean of likelihood values for binary predictions.

```python
def ll(x, p, epsilon=1e-9):
    """
    Calculate log-likelihood geometric mean.
    
    Parameters
    ----------
    x : array-like
        True binary labels (0 or 1)
    p : array-like
        Predicted probabilities for positive class
    epsilon : float, default=1e-9
        Smoothing factor for numerical stability
        
    Returns
    -------
    float
        Geometric mean of likelihood values
    """
```

**Mathematical Formula:**

$$
\text{LL}(x, p) = \exp\left(\frac{1}{n}\sum_{i=1}^{n} \left[x_i \log(p_i + \epsilon) + (1-x_i)\log(1-p_i + \epsilon)\right]\right)
$$

**Why Geometric Mean?**
- More robust to outliers than arithmetic mean
- Natural measure for likelihood values (multiplicative quantities)
- Handles probability products effectively

**Numerical Stability:**
- `epsilon = 1e-9` prevents log(0) errors
- Working in log-space avoids underflow
- Exp at the end recovers the geometric mean

#### `minimize_me(p, a)`

**Optimization Objective Function**

Helper function for finding the probability corresponding to a given likelihood.

```python
def minimize_me(p, a):
    """
    Objective function for information weight optimization.
    
    Parameters
    ----------
    p : float
        Probability value being optimized
    a : float
        Target likelihood value from ll()
        
    Returns
    -------
    float
        Absolute difference (objective to minimize)
    """
```

**Mathematical Formula:**

$$
f(p, a) = \left| p \log(p) + (1-p)\log(1-p) - \log(a) \right|
$$

This is the **binary entropy** equation solved for the probability that produces likelihood `a`.

#### `get_w(a, guess=0.5, bounds=[(0.5, 0.999)], tolerance=1e-09)`

**Information Weight Calculation**

Computes the information weight corresponding to a likelihood value by solving an optimization problem.

```python
def get_w(a, guess=0.5, bounds=[(0.5, 0.999)], tolerance=1e-09):
    """
    Compute information weight from likelihood value.
    
    Parameters
    ----------
    a : float
        Likelihood value from ll() function
    guess : float, default=0.5
        Initial guess for L-BFGS-B optimizer
    bounds : list of tuples, default=[(0.5, 0.999)]
        Probability bounds for optimization
    tolerance : float, default=1e-09
        Optimization tolerance (gtol)
        
    Returns
    -------
    float
        Information weight in range [0.5, 0.999]
    """
```

**Process:**
1. Takes likelihood `a` from `ll()`
2. Solves for probability `p` where entropy equals log-likelihood
3. Uses L-BFGS-B optimization with tight tolerance
4. Returns probability weight representing information content

**Interpretation:**
- `w = 0.5`: No information (random guessing)
- `w → 1.0`: Perfect information (certainty)
- Higher weight = more information in predictions

#### `calculate_imv(y_basic, y_enhanced, y, epsilon=1e-9)`

**IMV Score Calculation**

Main function for computing the relative information gain between two models.

```python
def calculate_imv(y_basic, y_enhanced, y, epsilon=1e-9):
    """
    Calculate IMV score comparing two model variants.
    
    Parameters
    ----------
    y_basic : array-like
        Predicted probabilities from baseline/null model
    y_enhanced : array-like
        Predicted probabilities from enhanced/full model
    y : array-like
        True binary labels
    epsilon : float, default=1e-9
        Smoothing factor for numerical stability
        
    Returns
    -------
    float
        IMV score (relative information gain)
    """
```

**Mathematical Formula:**

$$
\text{IMV} = \frac{w_{\text{enhanced}} - w_{\text{basic}}}{w_{\text{basic}}}
$$

Where:
- $w_{\text{basic}} = \text{get\_w}(\text{ll}(y, p_{\text{basic}}))$
- $w_{\text{enhanced}} = \text{get\_w}(\text{ll}(y, p_{\text{enhanced}}))$

**Interpretation:**
- **IMV > 0**: Enhanced model has more information (features help)
- **IMV ≈ 0**: Models are equivalent (features don't help)
- **IMV < 0**: Basic model better (rare, suggests overfitting)

**Example Values:**
- IMV = 0.05 → 5% information gain
- IMV = 0.20 → 20% information gain
- IMV = 0.50 → 50% information gain

#### `imv_from_probs(p_basic, p_enhanced, y, epsilon=1e-9)`

**Convenience Alias**

Direct alias for `calculate_imv()` with clearer parameter names.

---

## Module: imv/binary.py

### Purpose

Implements binary classification IMV with SHAP (Shapley value) attribution for feature importance analysis.

### Class: `BinaryIMV`

**Main Features:**
- K-fold cross-validation or train/test split
- Shapley value calculation for each feature
- Comprehensive visualization suite
- Integration with any scikit-learn compatible model

### Key Attributes

```python
class BinaryIMV:
    def __init__(
        self,
        data,                              # DataFrame with features and outcome
        outcome_variable,                  # Target column name
        optional_explanatory_variables,    # List of feature columns
        model_creator,                     # Function returning ML model
        split_method='kfold',              # 'kfold' or 'train_test'
        n_splits=5,                        # Number of CV folds
        prop_test=0.2,                     # Test set proportion
        model_type='classification',       # 'classification' or 'regression'
        random_state=None                  # Random seed
    ):
```

### Core Methods

#### `run_evaluation()`

Executes the full IMV evaluation pipeline with Shapley value attribution.

**Process:**
1. Split data (k-fold or train/test)
2. For each feature subset:
   - Train null model (intercept only)
   - Train enhanced model (with features)
   - Calculate IMV for feature contribution
3. Compute Shapley values using coalitional game theory
4. Aggregate results across folds

**Returns:**
- DataFrame with feature IMV scores and Shapley values

#### `evaluate_imvshapley()`

Alternative evaluation method with different feature selection strategy.

#### `calculate_imv_score(y_basic, y_enhanced, y)`

**Instance method wrapper** for `core.calculate_imv()`.

```python
def calculate_imv_score(self, y_basic, y_enhanced, y):
    """Calculate IMV using shared core function."""
    return calculate_imv(y_basic, y_enhanced, y)
```

### Visualization Methods

#### `plot_shapley_values(results, figsize=(10, 6))`

Bar plot of Shapley values showing feature importance.

#### `plot_imv_heatmap(results, figsize=(12, 8))`

Heatmap showing IMV contributions by feature and fold.

#### `plot_feature_importance(results, figsize=(10, 6))`

Combined visualization of mean IMV and standard deviation.

#### `plot_coalition_analysis(results, figsize=(12, 8))`

Detailed analysis of feature coalitions and interactions.

### Usage Example

```python
from imv import BinaryIMV
from sklearn.linear_model import LogisticRegression

# Define model creator
def create_model():
    return LogisticRegression(max_iter=1000)

# Initialize evaluator
evaluator = BinaryIMV(
    data=df,
    outcome_variable='target',
    optional_explanatory_variables=['age', 'income', 'education'],
    model_creator=create_model,
    split_method='kfold',
    n_splits=5
)

# Run evaluation
results = evaluator.run_evaluation()

# Visualize
fig = evaluator.plot_shapley_values(results)
```

### Shapley Value Calculation

**Coalitional Game Theory Approach:**

For each feature $i$, the Shapley value is:

$$
\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} [v(S \cup \{i\}) - v(S)]
$$

Where:
- $N$ = set of all features
- $S$ = coalition (subset) of features
- $v(S)$ = IMV value for coalition $S$
- Weights ensure fair attribution

**Implementation:**
1. Generate all feature subsets (power set)
2. Calculate IMV for each subset
3. Compute weighted marginal contributions
4. Aggregate using Shapley formula

---

## Module: imv/multiclass.py

### Purpose

Extends IMV to multiclass classification problems using pairwise comparisons and one-vs-all strategies.

### Class: `MulticlassIMV`

**Main Features:**
- Pairwise IMV matrices (class i vs class j)
- One-vs-all IMV scores
- K-fold cross-validation
- Heatmap and box plot visualizations

### Key Methods

#### `multinominal_imv_matrix(data, outcome_variable, p_base, p_enhanced)`

Creates pairwise IMV confusion matrix for all class combinations.

**Process:**
1. For each class pair (i, j):
   - Filter data to only classes i and j
   - Normalize probabilities for binary comparison
   - Calculate IMV(i vs j)
2. Construct matrix with IMV values

**Returns:**
- DataFrame: n_classes × n_classes matrix
- Element (i,j) = IMV discriminating class i from class j
- Diagonal is zero (no discrimination within same class)

**Interpretation:**
- High IMV(i,j): Features help distinguish class i from class j
- Low IMV(i,j): Little information for this pair
- Matrix generally asymmetric: IMV(i,j) ≠ IMV(j,i)

#### `k_fold_imv_matrix(data, outcome_variable, features, k=5)`

Computes average pairwise IMV matrix across k-fold CV.

**Benefits:**
- Uses all data for training and evaluation
- Reduces variance from single split
- Out-of-sample predictions
- More robust estimates

#### `k_fold_one_vs_all(data, outcome_variable, features, k=5)`

One-vs-all approach: each class vs all others combined.

**Process:**
1. For each class:
   - Treat as binary problem (class vs rest)
   - Calculate IMV using k-fold CV
2. Return DataFrame with IMV per class

**Use Case:**
- Simpler than pairwise comparisons
- Good for identifying which classes benefit most from features
- Faster computation for many classes

### Visualization Methods

#### `visualize_imv_matrix(imv_matrix, figsize=(10, 8))`

Heatmap of pairwise IMV matrix.

**Features:**
- Annotated cells with IMV values
- Color scale showing information gain
- Class labels on axes

#### `plot_ova_results(ova_results, figsize=(10, 6))`

Bar plot of one-vs-all IMV scores.

#### `plot_multi_imv_distributions(imv_results, figsize=(10, 6))`

Box plots showing IMV distributions across outcomes.

### Usage Example

```python
from imv import MulticlassIMV

# Initialize
evaluator = MulticlassIMV()

# Pairwise IMV matrix
imv_matrix = evaluator.k_fold_imv_matrix(
    data=df,
    outcome_variable='species',
    features=['sepal_length', 'sepal_width'],
    k=5
)

# Visualize
fig = evaluator.visualize_imv_matrix(imv_matrix)

# One-vs-all
ova_results = evaluator.k_fold_one_vs_all(
    data=df,
    outcome_variable='species',
    features=['sepal_length', 'sepal_width'],
    k=5
)
```

---

## Module: imv/ablation.py

### Purpose

Applies IMV to deep learning models (especially transformers) to quantify the information contribution of architectural components through ablation studies.

### Class: `AblationIMV`

**Main Features:**
- Automatic GPU detection (CUDA > MPS > CPU)
- Transformer layer reduction
- Training and evaluation pipeline
- IMV matrices for model comparisons

### Key Attributes

```python
class AblationIMV:
    def __init__(self, random_seed=42):
        """
        Initialize with automatic device detection.
        
        Device Priority:
        1. CUDA (NVIDIA GPUs)
        2. MPS (Apple Silicon)
        3. CPU (fallback)
        """
```

### Device Management

```python
# Automatic detection
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

### Core Methods

#### `reduce_bert_layers(model, num_layers_to_keep)`

Performs layer ablation by removing transformer layers.

**Parameters:**
- `model`: DistilBERT or similar transformer
- `num_layers_to_keep`: Layers to keep from beginning

**Process:**
1. Access model's transformer layers
2. Slice to keep only first N layers
3. Return modified model (in-place)

**Use Case:**
- Measure importance of model depth
- Study layer-wise contributions
- Identify diminishing returns

#### `train_and_evaluate(model, train_dataloader, test_dataloader, ...)`

Complete training and evaluation pipeline.

**Parameters:**
- Learning rate, epochs, warmup steps
- Scheduler configuration
- Progress tracking with tqdm

**Returns:**
- Trained model
- Training history
- Test predictions
- Performance metrics

#### `calculate_imv_matrix(predictions_dict, y, prob_column='prob_positive')`

Creates IMV matrix comparing different model variants.

**Process:**
1. For each model pair (i, j):
   - Use model i as "enhanced"
   - Use model j as "basic"
   - Calculate IMV(i vs j)
2. Construct comparison matrix

**Use Case:**
- Compare 6-layer vs 3-layer model
- Quantify information loss from ablation
- Identify critical components

#### `average_imv_matrices(matrices_list)`

Averages IMV matrices across multiple runs.

**Benefits:**
- Reduces random variation
- More stable estimates
- Statistical confidence

### Usage Example

```python
from imv import AblationIMV
from transformers import DistilBertForSequenceClassification

# Initialize
ablation = AblationIMV(random_seed=42)

# Create models with different depths
model_6layer = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)

model_3layer = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)
model_3layer = ablation.reduce_bert_layers(model_3layer, 3)

# Train both models
model_6layer = ablation.train_and_evaluate(
    model_6layer, train_dl, test_dl, epochs=3
)
model_3layer = ablation.train_and_evaluate(
    model_3layer, train_dl, test_dl, epochs=3
)

# Get predictions
predictions = {
    '6-layer': predict(model_6layer, test_dl),
    '3-layer': predict(model_3layer, test_dl)
}

# Calculate IMV
imv_matrix = ablation.calculate_imv_matrix(
    predictions, y_test, prob_column='prob_positive'
)
```

---

## Mathematical Foundation

### Information Theory Basis

IMV is grounded in **information theory** and **statistical decision theory**.

#### 1. Shannon Entropy

Binary entropy for probability $p$:

$$
H(p) = -p \log_2(p) - (1-p)\log_2(1-p)
$$

- Measures uncertainty/information content
- Maximum at $p = 0.5$ (maximum uncertainty)
- Minimum at $p = 0, 1$ (certainty)

#### 2. Kullback-Leibler Divergence

Information gain from model predictions:

$$
D_{KL}(P \| Q) = \sum_i P(i) \log\frac{P(i)}{Q(i)}
$$

Where:
- $P$ = enhanced model distribution
- $Q$ = baseline model distribution

IMV approximates this divergence using likelihood ratios.

#### 3. Geometric Mean Likelihood

Why geometric mean for likelihood?

**Arithmetic Mean:**
$$
\bar{L}_{\text{arithmetic}} = \frac{1}{n}\sum_{i=1}^{n} L_i
$$

**Geometric Mean:**
$$
\bar{L}_{\text{geometric}} = \left(\prod_{i=1}^{n} L_i\right)^{1/n} = \exp\left(\frac{1}{n}\sum_{i=1}^{n} \log L_i\right)
$$

**Advantages of Geometric Mean:**
- Natural for multiplicative quantities (probabilities)
- More robust to outliers
- Relates directly to entropy
- Preserves likelihood ordering

### IMV Derivation

**Step 1: Likelihood**

For binary outcome $y \in \{0, 1\}$ and prediction $p$:

$$
L(y, p) = p^y (1-p)^{1-y}
$$

**Step 2: Geometric Mean Likelihood**

$$
\bar{L} = \exp\left(\frac{1}{n}\sum_{i=1}^{n} \left[y_i \log p_i + (1-y_i)\log(1-p_i)\right]\right)
$$

**Step 3: Information Weight**

Solve for $w$ where binary entropy equals log-likelihood:

$$
-w\log w - (1-w)\log(1-w) = \log \bar{L}
$$

This gives weight $w \in [0.5, 1)$ representing information content.

**Step 4: IMV Score**

Relative information gain:

$$
\text{IMV} = \frac{w_{\text{enhanced}} - w_{\text{basic}}}{w_{\text{basic}}}
$$

### Properties

1. **Scale Invariant**: IMV is unitless (percentage gain)
2. **Bounded Below**: IMV > -1 (can't lose all information)
3. **Unbounded Above**: IMV can be arbitrarily large
4. **Additive**: For independent features, IMVs approximately add

---

## Implementation Details

### Optimization Strategy

#### L-BFGS-B Algorithm

Used in `get_w()` for finding information weights.

**Why L-BFGS-B?**
- Limited-memory (efficient for single variable)
- Handles bounds naturally
- Quasi-Newton method (fast convergence)
- No gradient needed (uses finite differences)

**Configuration:**
```python
res = minimize(
    minimize_me,
    guess=0.5,
    args=a,
    method='L-BFGS-B',
    bounds=[(0.5, 0.999)],
    options={'ftol': 0, 'gtol': 1e-09}
)
```

**Tolerances:**
- `ftol=0`: No function value tolerance (force gradient check)
- `gtol=1e-09`: Very tight gradient tolerance (high precision)

### Numerical Stability

#### Epsilon Smoothing

```python
epsilon = 1e-9
log_term = np.log(p + epsilon)
```

**Purpose:**
- Prevents `log(0)` → `-inf`
- Prevents division by zero
- Minimal impact on valid probabilities (p > 0.001)

#### Probability Bounds

```python
bounds = [(0.5, 0.999)]
```

**Rationale:**
- Lower bound 0.5: No information = random guessing
- Upper bound 0.999: Prevent numerical issues near 1.0
- Allows 99.9% certainty (sufficient for practical use)

### Performance Optimization

#### Vectorization

Core functions use NumPy vectorization:

```python
# Vectorized likelihood calculation
z = (np.log(p + epsilon) * x) + (np.log(1 - p + epsilon) * (1 - x))
return np.exp(np.sum(z) / len(z))
```

**Benefits:**
- 10-100x faster than Python loops
- Leverages BLAS/LAPACK
- Cache-friendly memory access

#### Memory Efficiency

- Uses generators for power set enumeration
- Streams predictions in batches
- Clears intermediate arrays

#### Parallel Processing

Potential for parallelization:
- Shapley value calculations (independent coalitions)
- K-fold cross-validation (independent folds)
- Pairwise IMV comparisons (independent pairs)

---

## Usage Examples

### Example 1: Binary Classification with Shapley Values

```python
from imv import BinaryIMV
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Load data
df = pd.read_csv('adult_income.csv')

# Define model creator
def create_model():
    return LogisticRegression(max_iter=1000, random_state=42)

# Initialize evaluator
evaluator = BinaryIMV(
    data=df,
    outcome_variable='income',
    optional_explanatory_variables=['age', 'education', 'hours_per_week'],
    model_creator=create_model,
    split_method='kfold',
    n_splits=5,
    random_state=42
)

# Run evaluation
results = evaluator.run_evaluation()

# Print results
print(results[['feature', 'imv', 'shapley_value']])

# Visualize
fig = evaluator.plot_shapley_values(results, figsize=(10, 6))
fig.savefig('shapley_values.png')
```

### Example 2: Multiclass Classification

```python
from imv import MulticlassIMV
from sklearn.datasets import load_iris
import pandas as pd

# Load iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Initialize evaluator
evaluator = MulticlassIMV()

# Pairwise IMV matrix
imv_matrix = evaluator.k_fold_imv_matrix(
    data=df,
    outcome_variable='species',
    features=['sepal length (cm)', 'sepal width (cm)'],
    k=5,
    random_state=42
)

# Visualize pairwise comparisons
fig = evaluator.visualize_imv_matrix(imv_matrix, figsize=(10, 8))

# One-vs-all approach
ova_results = evaluator.k_fold_one_vs_all(
    data=df,
    outcome_variable='species',
    features=['sepal length (cm)', 'sepal width (cm)'],
    k=5,
    random_state=42
)

# Visualize one-vs-all
fig2 = evaluator.plot_ova_results(ova_results, figsize=(10, 6))
```

### Example 3: Transformer Ablation Study

```python
from imv import AblationIMV
from transformers import (
    DistilBertForSequenceClassification,
    AutoTokenizer,
    get_scheduler
)
from torch.optim import AdamW
from torch.utils.data import DataLoader

# Initialize
ablation = AblationIMV(random_seed=42)

# Prepare data
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
train_dl = DataLoader(train_dataset, batch_size=16)
test_dl = DataLoader(test_dataset, batch_size=16)

# Create models with different layers
models = {}
for n_layers in [2, 4, 6]:
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2
    )
    if n_layers < 6:
        model = ablation.reduce_bert_layers(model, n_layers)
    
    # Train
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = get_scheduler("linear", optimizer, 
                             num_warmup_steps=100,
                             num_training_steps=1000)
    
    model = ablation.train_and_evaluate(
        model, train_dl, test_dl,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=3
    )
    
    models[f'{n_layers}-layer'] = model

# Get predictions for all models
predictions = {}
for name, model in models.items():
    preds = predict_with_model(model, test_dl)
    predictions[name] = preds

# Calculate IMV matrix
imv_matrix = ablation.calculate_imv_matrix(
    predictions,
    y_test,
    prob_column='prob_positive'
)

print("\nIMV Matrix (row vs column):")
print(imv_matrix)
```

### Example 4: Direct Core Function Usage

```python
from imv import ll, get_w, calculate_imv
import numpy as np

# True labels
y = np.array([1, 0, 1, 1, 0, 1, 0])

# Baseline predictions (intercept only)
p_basic = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

# Enhanced predictions (with features)
p_enhanced = np.array([0.8, 0.2, 0.75, 0.85, 0.3, 0.9, 0.15])

# Calculate likelihoods
ll_basic = ll(y, p_basic)
ll_enhanced = ll(y, p_enhanced)

print(f"Baseline likelihood: {ll_basic:.6f}")
print(f"Enhanced likelihood: {ll_enhanced:.6f}")

# Calculate information weights
w_basic = get_w(ll_basic)
w_enhanced = get_w(ll_enhanced)

print(f"Baseline weight: {w_basic:.6f}")
print(f"Enhanced weight: {w_enhanced:.6f}")

# Calculate IMV
imv = calculate_imv(p_basic, p_enhanced, y)

print(f"IMV: {imv:.6f} ({imv*100:.2f}% information gain)")
```

---

## Performance Considerations

### Computational Complexity

#### Binary IMV with Shapley Values

- **Power set enumeration**: $O(2^n)$ where n = number of features
- **Model training per subset**: $O(n_{\text{samples}} \times n_{\text{features}})$
- **Total**: $O(2^n \times n_{\text{samples}} \times n_{\text{features}})$

**Practical Limits:**
- Works well for n ≤ 10 features
- Becomes expensive for n > 15 features
- Consider sampling coalitions for large n

#### Multiclass IMV

- **Pairwise comparisons**: $O(n_{\text{classes}}^2)$
- **K-fold CV**: $O(k \times n_{\text{samples}})$
- **Total**: $O(k \times n_{\text{classes}}^2 \times n_{\text{samples}})$

**Scaling:**
- Linear in sample size
- Quadratic in number of classes
- Well-suited for moderate class counts (2-20)

#### Ablation IMV

- **Model training**: $O(n_{\text{epochs}} \times n_{\text{samples}} \times n_{\text{params}})$
- **Dominated by**: Neural network training time
- **GPU acceleration**: 10-100x speedup

### Memory Requirements

- **Binary IMV**: Stores results for all coalitions: $O(2^n)$
- **Multiclass IMV**: Stores pairwise matrix: $O(n_{\text{classes}}^2)$
- **Ablation IMV**: Stores model parameters: $O(n_{\text{params}})$

### Optimization Strategies

1. **Parallel Coalition Evaluation**: Use `joblib` or `multiprocessing`
2. **Early Stopping**: Skip coalitions with low expected contribution
3. **Batch Predictions**: Process multiple samples together
4. **GPU Utilization**: For deep learning ablation studies
5. **Caching**: Store and reuse model predictions

### Recommended Hardware

- **Binary IMV (< 10 features)**: Standard CPU, 4GB RAM
- **Binary IMV (> 10 features)**: Multi-core CPU, 8GB+ RAM
- **Multiclass IMV**: Standard CPU, 4GB RAM
- **Ablation IMV**: GPU (CUDA or MPS), 16GB+ RAM

---

## Backward Compatibility

### Old Class Names

The package maintains backward compatibility with old class names:

```python
# Old imports still work
from imv import IMVEvaluator      # → BinaryIMV
from imv import MultinomialIMV    # → MulticlassIMV

# New recommended imports
from imv import BinaryIMV
from imv import MulticlassIMV
```

### Migration Guide

**Old Code:**
```python
from metrics.shap_imv import IMVEvaluator

evaluator = IMVEvaluator(data, ...)
results = evaluator.run_evaluation()
```

**New Code:**
```python
from imv import BinaryIMV

evaluator = BinaryIMV(data, ...)
results = evaluator.run_evaluation()
```

All method signatures remain unchanged for seamless migration.

---

## References

1. **Shapley Values**: Shapley, L. S. (1953). "A value for n-person games"
2. **Information Theory**: Shannon, C. E. (1948). "A Mathematical Theory of Communication"
3. **IMV Method**: [Original paper reference]
4. **SHAP**: Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions"

---

## Contributing

See the main README.md for contribution guidelines.

## License

See LICENSE file in the repository root.

---

*Last Updated: November 30, 2024*
*Package Version: 1.0.0*
