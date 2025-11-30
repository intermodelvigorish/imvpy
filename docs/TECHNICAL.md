# Technical Documentation: IMV Package

## Table of Contents
1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Module: metrics/shap_imv.py](#module-metricsshap_imvpy)
4. [Module: metrics/multi_imv.py](#module-metricsmulti_imvpy)
5. [Module: ablation/ablate_imv.py](#module-ablationablate_imvpy)
6. [Mathematical Foundations](#mathematical-foundations)
7. [Implementation Details](#implementation-details)
8. [Performance Considerations](#performance-considerations)

---

## Overview

The IMV (Information Model Vigor) package provides a framework for quantifying feature importance and model performance using information-theoretic principles. The package consists of three main modules:

- **SHAP-IMV** (`metrics/shap_imv.py`): Binary classification with Shapley value attribution
- **Multi-class IMV** (`metrics/multi_imv.py`): Multi-class classification (3+ classes)
- **Ablation IMV** (`ablation/ablate_imv.py`): Deep learning ablation studies

### Key Innovation

Traditional metrics (accuracy, F1) measure prediction correctness but not information content. IMV quantifies how much information a model captures about the true labels, enabling:
- Fair comparison between models with different architectures
- Feature importance without model-specific assumptions
- Ablation study quantification in deep learning

---

## Core Concepts

### 1. Information Weight (w)

The information weight represents the "certainty" or "information content" of predictions:

```python
w = argmin_p |p*log(p) + (1-p)*log(1-p) - log(LL)|
```

Where:
- `p` ∈ [0.5, 0.999]: Probability weight
- `LL`: Log-likelihood of predictions
- Result: Higher `w` means more information

### 2. IMV Score

IMV measures relative information gain:

```python
IMV = (w_enhanced - w_basic) / w_basic
```

**Interpretation:**
- `IMV = 0.10` → Enhanced model has 10% more information
- `IMV = 0` → Models are equivalent
- `IMV < 0` → Enhanced model is worse (rare)

### 3. SHAP-IMV Values

Shapley values applied to IMV provide fair feature attribution:

```python
SHAP-IMV(v) = Σ [weight(|S|, n) * (IMV(S ∪ {v}) - IMV(S))]
```

Where:
- `v`: Feature being evaluated
- `S`: All subsets not containing `v`
- `weight`: Shapley weight based on coalition size

---

## Module: metrics/shap_imv.py

### Architecture

```
IMVEvaluator
├── Core Computations
│   ├── ll()           # Log-likelihood geometric mean
│   ├── get_w()        # Information weight via optimization
│   └── calculate_imv() # Relative information gain
├── Model Evaluation
│   ├── compute_imv_method()  # Single feature combination
│   └── run_evaluation()      # All combinations (parallel)
├── Shapley Attribution
│   ├── calculate_weight()           # Shapley weights
│   ├── calculate_imvshapley_value() # Per-feature SHAP-IMV
│   └── evaluate_imvshapley()        # All features + visualization
└── Visualization
    └── plot_single_var_combinations_layered_violin_centralized_zero()
```

### Workflow

1. **Initialization**
   ```python
   evaluator = IMVEvaluator(
       data=df,
       outcome_variable='target',
       optional_explanatory_variables=['age', 'income', 'education'],
       model_creator=lambda: LogisticRegression(),
       split_method='kfold',
       n_splits=5,
       model_type='classification'
   )
   ```

2. **Compute All Combinations**
   ```python
   evaluator.run_evaluation()
   # Computes 2^n IMV scores where n = number of features
   # Uses joblib Parallel with n_jobs=-1 for speed
   ```

3. **Shapley Attribution**
   ```python
   evaluator.evaluate_imvshapley()
   # Computes SHAP-IMV for each feature
   # Creates visualization
   ```

### Key Implementation Details

#### Parallel Processing
```python
with tqdm_joblib(tqdm(desc="Evaluating IMV combinations", total=len(combinations_list))):
    parallel_results = Parallel(n_jobs=-1)(
        delayed(self.compute_imv_method)(subset) for subset in combinations_list
    )
```
- Uses all CPU cores (`n_jobs=-1`)
- Progress bar via `tqdm_joblib`
- Each combination evaluated independently

#### Cross-Validation
```python
if self.split_method == 'kfold':
    kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_seed)
    for train_index, test_index in kf.split(X):
        # Train null model (constant only)
        model_basic.fit(X_train[['constant']], y_train)
        # Train enhanced model (with features)
        model_enhanced.fit(X_train, y_train)
        # Compute IMV on test set
```

#### Optimization Precision
```python
res = minimize(cls.minimize_me, guess, args=(a,), 
               options={'ftol': 0, 'gtol': 1e-09},
               method='L-BFGS-B', bounds=[(0.5, 0.999)])
```
- L-BFGS-B for bounded optimization
- Tight gradient tolerance (1e-09)
- Bounds prevent edge cases (p=0.5 or p=1)

---

## Module: metrics/multi_imv.py

### Architecture

```
MultinomialIMV
├── Core Computations (Binary)
│   ├── ll()          # Log-likelihood (binary)
│   ├── get_w()       # Information weight
│   └── minimize_me() # Optimization objective
├── One-vs-All
│   ├── one_vs_all_single_fold()  # Single fold, all classes
│   └── k_fold_one_vs_all()       # K-fold cross-validation
├── Pairwise Matrix
│   ├── multinominal_imv_matrix() # Single fold IMV matrix
│   └── k_fold_imv_matrix()       # K-fold average
└── Visualization
    ├── multinomial_IMV_heatmap()  # Pairwise confusion matrix
    └── multinomial_IMV_boxplot()  # One-vs-all distribution
```

### Two Evaluation Approaches

#### 1. One-vs-All IMV
Treats each class as positive vs all others combined:

```python
imv_results, imv_average = evaluator.k_fold_one_vs_all()
# Returns:
# - imv_results: (n_folds, n_classes) array
# - imv_average: (n_classes,) mean across folds
```

**Use Case:** Which class is most distinguishable from others?

#### 2. Pairwise IMV Matrix
Computes IMV for every pair of classes:

```python
matrices_list, matrix_avg = evaluator.k_fold_imv_matrix()
# Returns:
# - matrices_list: List of (n_classes, n_classes) matrices
# - matrix_avg: Average matrix across folds
```

**Use Case:** Which class pairs are most/least distinguishable?

### Key Implementation Details

#### Probability Normalization
For pairwise comparison of classes i and j:
```python
# Filter to only these two classes
mask = data[outcome_variable].isin([outcome_i, outcome_j])

# Normalize probabilities for binary comparison
p_b = p_base[mask, i] / np.sum(p_base[mask][:, [i, j]], axis=1)
p_e = p_enhanced[mask, i] / np.sum(p_enhanced[mask][:, [i, j]], axis=1)
```
This converts multi-class probabilities to binary probabilities for IMV calculation.

#### Index Offset Handling
```python
# Determine if outcomes start at 0 or 1
index_offset = 0 if outcomes[0] == 0 else 1

for outcome in outcomes:
    # Access probability array with correct index
    p_b = p_base[:, outcome - index_offset]
```
Handles datasets with 0-indexed classes (0, 1, 2) vs 1-indexed (1, 2, 3).

#### Null Model
```python
# Base model uses only constant (intercept)
X_train_constant = np.ones((X_train.shape[0], 1))
model_basic.fit(X_train_constant, y_train)
```
Multi-class null model predicts class frequencies.

---

## Module: ablation/ablate_imv.py

### Architecture

```
AblationIMV
├── Device Management
│   ├── __init__()     # Auto-detect CUDA/MPS/CPU
│   └── set_seed()     # Reproducibility across devices
├── Core IMV Computations
│   ├── ll()           # Log-likelihood
│   ├── get_w()        # Information weight (high precision)
│   └── calculate_imv() # IMV score
├── Model Ablation
│   ├── reduce_bert_layers()     # Layer reduction
│   └── train_and_evaluate()     # Train + metrics
├── Comparative Analysis
│   ├── calculate_imv_matrix()   # Pairwise model comparison
│   └── average_imv_matrices()   # Multi-seed averaging
└── [Extensible for other ablations]
```

### Hardware Support

#### Automatic Device Detection
```python
# Priority: CUDA > MPS > CPU
if torch.cuda.is_available():
    self.device = torch.device("cuda")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    self.device = torch.device("mps")  # Apple Silicon
else:
    self.device = torch.device("cpu")
```

**Supported Platforms:**
- NVIDIA GPUs (CUDA)
- Apple M1/M2/M3 (MPS) - Requires PyTorch ≥ 1.12
- CPU fallback

#### Reproducibility Across Devices
```python
def set_seed(self, seed=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
```

### Ablation Study Workflow

1. **Setup Ablation Variants**
   ```python
   ablator = AblationIMV(random_seed=42)
   
   # Create model variants
   model_full = DistilBertForSequenceClassification.from_pretrained(
       "distilbert-base-uncased", num_labels=2
   )
   model_4layer = ablator.reduce_bert_layers(model_full, 4)
   model_2layer = ablator.reduce_bert_layers(model_full, 2)
   ```

2. **Train Each Variant**
   ```python
   results = {}
   for name, model in [('6-layer', model_full), ('4-layer', model_4layer)]:
       result = ablator.train_and_evaluate(
           model, train_loader, test_loader,
           num_epochs=3, lr=2e-5, seed=42
       )
       results[name] = result['test_predictions']
   ```

3. **Compute IMV Matrix**
   ```python
   imv_matrix = ablator.calculate_imv_matrix(results)
   # Shows pairwise IMV comparisons
   ```

4. **Multi-Seed Averaging**
   ```python
   matrices = []
   for seed in [42, 123, 456]:
       # Repeat study with different seed
       imv_mat = run_ablation_study(seed)
       matrices.append(imv_mat)
   
   avg_matrix = ablator.average_imv_matrices(matrices)
   ```

### Key Implementation Details

#### High-Precision Optimization
```python
res = minimize(AblationIMV.minimize_me, guess, args=a,
               options={'ftol': 0, 'gtol': 1e-20},  # Note: 1e-20 vs 1e-09
               method='L-BFGS-B', bounds=bounds)
```
Uses tighter tolerance than metrics modules for precise deep learning comparisons.

#### Training Loop with Device Transfer
```python
for batch in train_dataloader:
    # Move batch to GPU/CPU
    batch = {k: v.to(self.device) for k, v in batch.items()}
    
    outputs = model(**batch)
    loss = outputs.loss
    
    # Standard PyTorch training
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

#### Layer Ablation
```python
@staticmethod
def reduce_bert_layers(model, num_layers_to_keep):
    # Slice ModuleList to keep first N layers
    model.distilbert.transformer.layer = torch.nn.ModuleList(
        model.distilbert.transformer.layer[:num_layers_to_keep]
    )
    return model
```
Modifies model in-place by removing later layers.

---

## Mathematical Foundations

### Log-Likelihood Geometric Mean

**Formula:**
```
LL(x, p) = exp(mean(x*log(p + ε) + (1-x)*log(1-p + ε)))
```

**Properties:**
- Range: (0, 1)
- Higher = better predictions
- ε = 1e-9 for numerical stability

**Why Geometric Mean?**
- Multiplicative nature of probabilities
- Less sensitive to outliers than arithmetic mean
- Information-theoretic interpretation

### Information Weight Optimization

**Objective:**
```
minimize |p*log(p) + (1-p)*log(1-p) - log(LL)|
```

**Entropy Connection:**
The left term `p*log(p) + (1-p)*log(1-p)` is binary entropy. We're finding the probability `p` whose entropy matches the observed likelihood.

**Why [0.5, 0.999] Bounds?**
- 0.5: Random guessing (no information)
- 0.999: Near-perfect prediction (avoid log(0))
- Bounded optimization more stable than unconstrained

### Shapley Values

**Formula:**
```
φ(v) = Σ_S [|S|!(n-|S|-1)! / n!] * [V(S ∪ {v}) - V(S)]
```

Where:
- `φ(v)`: Shapley value for feature v
- `S`: Subset not containing v
- `V(·)`: Characteristic function (IMV)
- Weight: Depends on coalition size

**Properties:**
- **Efficiency:** Sum of all Shapley values = Total IMV
- **Symmetry:** If features contribute equally, equal Shapley values
- **Dummy:** If feature contributes nothing, Shapley value = 0
- **Additivity:** Shapley values are linear

---

## Implementation Details

### Memory Management

#### SHAP-IMV Power Set
For n features, computes 2^n combinations:
- n=5: 32 combinations (fast)
- n=10: 1024 combinations (manageable)
- n=15: 32,768 combinations (slow)
- n=20: 1,048,576 combinations (infeasible)

**Recommendation:** Use SHAP-IMV for n ≤ 12 features.

#### Multi-class IMV
For k classes:
- One-vs-All: k models per fold
- Pairwise Matrix: k*(k-1) comparisons per fold

**Memory:** O(k² * n_samples * n_folds)

#### Ablation IMV
Deep learning models are memory-intensive:
- DistilBERT: ~250MB
- BERT: ~400MB
- GPT-2: ~500MB

**Recommendation:** 
- Batch size 8-16 for 8GB GPU
- Batch size 32-64 for 16GB+ GPU
- Use gradient accumulation for larger effective batches

### Numerical Stability

#### Epsilon Smoothing
```python
epsilon = 1e-9
z = (np.log(p + epsilon) * x) + (np.log(1 - p + epsilon) * (1 - x))
```
Prevents `log(0)` when predictions are exactly 0 or 1.

#### Bounded Optimization
```python
bounds = [(0.5, 0.999)]
```
Prevents optimization from exploring extreme probabilities.

#### Loss Function Choice
For ablation studies, use:
- **CrossEntropyLoss**: Built into HuggingFace models
- **Focal Loss**: For imbalanced datasets
- **Label Smoothing**: Improves calibration

### Parallel Processing

#### CPU Parallelism (SHAP-IMV)
```python
Parallel(n_jobs=-1)(delayed(compute_imv_method)(subset) for subset in combinations)
```
- Uses all CPU cores
- Each combination is independent
- Linear speedup with cores

#### GPU Parallelism (Ablation IMV)
```python
# Data parallelism
model = torch.nn.DataParallel(model)

# Distributed training
torch.distributed.init_process_group(backend='nccl')
model = DistributedDataParallel(model)
```
Not implemented by default; add for multi-GPU setups.

---

## Performance Considerations

### Computational Complexity

| Module | Time Complexity | Space Complexity |
|--------|----------------|------------------|
| SHAP-IMV | O(2^n * k * m) | O(2^n) |
| Multi-class IMV (One-vs-All) | O(k * n_folds * m) | O(k * n_samples) |
| Multi-class IMV (Pairwise) | O(k² * n_folds * m) | O(k² * n_samples) |
| Ablation IMV | O(a * e * n_samples * d) | O(model_size) |

Where:
- n: Number of features
- k: Number of classes
- m: Model training time
- a: Number of ablation variants
- e: Training epochs
- d: Model forward/backward pass complexity

### Optimization Tips

#### 1. Reduce Feature Count (SHAP-IMV)
```python
# Feature selection before IMV
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Now run IMV on 10 features instead of 50
```

#### 2. Use Train-Test Split Instead of K-Fold
```python
# Faster but less stable
evaluator = IMVEvaluator(split_method='train_test_split', prop_test=0.2)
```

#### 3. Reduce Ablation Study Size
```python
# Instead of all layer counts, sample key points
layer_counts = [6, 4, 2, 1]  # Full, 2/3, 1/3, minimal
```

#### 4. Mixed Precision Training (Ablation)
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(**batch)
    loss = outputs.loss
scaler.scale(loss).backward()
```
2x speedup on modern GPUs.

#### 5. Early Stopping
```python
# Stop training if validation loss plateaus
if val_loss improvement < threshold for patience epochs:
    break
```

### Monitoring

#### Progress Tracking
All modules use `tqdm` for progress:
```python
from tqdm import tqdm
for epoch in tqdm(range(num_epochs), desc="Training"):
    # ...
```

#### Memory Profiling
```python
import torch
print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

#### Time Profiling
```python
import time
start = time.time()
evaluator.run_evaluation()
print(f"Time elapsed: {time.time() - start:.2f} seconds")
```

---

## Best Practices

### 1. Feature Scaling
IMV is scale-invariant for linear models but helps with convergence:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 2. Random Seed Management
Always set seeds for reproducibility:
```python
evaluator = IMVEvaluator(random_seed=42)
ablator = AblationIMV(random_seed=42)
```

### 3. Model Selection
- **SHAP-IMV**: Works with any sklearn-compatible model
- **Multi-class IMV**: Requires `predict_proba()` method
- **Ablation IMV**: Designed for PyTorch models

### 4. Validation Strategy
- Use k-fold for small datasets (n < 1000)
- Use train-test split for large datasets (n > 10,000)
- Use stratified splits for imbalanced data

### 5. Interpretation
- IMV > 0.05: Meaningful information gain (5%+)
- IMV < 0.01: Negligible gain
- Negative IMV: Check for overfitting

---

## Troubleshooting

### Common Issues

#### 1. Optimization Fails to Converge
```python
# Symptom: "Optimization terminated unsuccessfully"
# Solution: Relax tolerance
res = minimize(..., options={'gtol': 1e-06})  # Instead of 1e-09
```

#### 2. Memory Error (Ablation)
```python
# Symptom: CUDA out of memory
# Solution: Reduce batch size
train_dataloader = DataLoader(dataset, batch_size=8)  # Instead of 32
```

#### 3. Slow SHAP-IMV
```python
# Symptom: Takes hours for 15 features
# Solution: Reduce features or use approximate methods
# Option 1: Feature selection
X_selected = select_top_k_features(X, y, k=10)

# Option 2: Sample combinations (future work)
# Not currently implemented
```

#### 4. Inconsistent Results Across Runs
```python
# Symptom: Different IMV values each time
# Solution: Set random seed everywhere
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
evaluator = IMVEvaluator(random_seed=42)
```

---

## Future Enhancements

### Planned Features
1. **Approximate SHAP-IMV**: Monte Carlo sampling for large feature sets
2. **GPU Acceleration**: CUDA kernels for IMV computation
3. **Model-Agnostic Ablation**: Support for TensorFlow, JAX
4. **Streaming IMV**: Online/incremental computation
5. **Confidence Intervals**: Bootstrap estimation of IMV uncertainty

### Extension Points
The package is designed for extensibility:
- Add new ablation types in `ablate_imv.py`
- Implement custom visualization methods
- Extend to regression tasks
- Add probabilistic IMV variants

---

## References

1. Valler, M., & Liu, J. (2024). "Information Model Vigor: A framework for feature importance."
2. Shapley, L. S. (1953). "A value for n-person games." Contributions to the Theory of Games.
3. Lundberg, S. M., & Lee, S. I. (2017). "A unified approach to interpreting model predictions." NIPS.
4. Devlin, J., et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding."

---

**Document Version:** 1.0  
**Last Updated:** November 30, 2025  
**Maintainer:** @intermodelvigorish
