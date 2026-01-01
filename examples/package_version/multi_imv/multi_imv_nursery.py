"""
Multi-IMV Nursery Analysis
Compares Logistic Regression, XGBoost, and LightGBM using Multi-IMV for multi-class classification
Replicates results from Figure 3 and performance tables of the paper
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, brier_score_loss
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from imv import MulticlassIMV

# Try to import XGBoost and LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available. Install with: pip install lightgbm")

# Setup directories
EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(EXAMPLE_DIR), 'data')
FIGURES_DIR = os.path.join(EXAMPLE_DIR, 'figures')
RESULTS_DIR = os.path.join(EXAMPLE_DIR, 'results')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_and_prepare_nursery_data():
    """
    Load and prepare the Nursery dataset
    Original: 5 classes (not_recommend, recommend, very_recommend, priority, spec_prior)
    Filtered: 3 classes (not_recommend, priority, spec_prior)
    Features: 8 categorical variables about family and nursery characteristics
    """
    data_file = os.path.join(DATA_DIR, 'nursery_processed.csv')
    
    # Try to load cached data
    if os.path.exists(data_file):
        try:
            df_final = pd.read_csv(data_file)
            print(f"Loaded processed dataset from: {data_file}")
            features = [col for col in df_final.columns if col != 'target']
            classes = sorted(df_final['target'].unique())
            print(f"Dataset shape: {df_final.shape}")
            print(f"Features: {features}")
            print(f"Classes: {classes}")
            print(f"Class distribution:\n{df_final['target'].value_counts().sort_index()}")
            return df_final, features, classes
        except Exception as e:
            print(f"Warning: failed to read cached dataset: {e}")
    
    try:
        from ucimlrepo import fetch_ucirepo
        
        print("Fetching Nursery dataset...")
        # Nursery dataset ID: 76
        nursery = fetch_ucirepo(id=76)
        
        X = nursery.data.features
        y = nursery.data.targets
        
        print(f"Original dataset shape: {X.shape}")
        print(f"Original features: {list(X.columns)}")
        
        # Prepare target variable
        if isinstance(y, pd.DataFrame):
            target_col = y.columns[0]
            y_series = y[target_col]
        else:
            y_series = y
        
        print(f"Original class distribution:\n{y_series.value_counts()}")
        
        # Filter to keep only 3 classes: not_recom, priority, spec_prior
        # Remove 'recommend' and 'very_recom' classes (limited instances)
        keep_classes = ['not_recom', 'priority', 'spec_prior']
        mask = y_series.isin(keep_classes)
        
        X_filtered = X[mask]
        y_filtered = y_series[mask]
        
        print(f"\nFiltered class distribution:\n{y_filtered.value_counts()}")
        
        # Encode categorical features as numeric
        from sklearn.preprocessing import LabelEncoder
        X_encoded = X_filtered.copy()
        
        for col in X_encoded.columns:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col])
        
        # Encode target variable
        # Map: not_recom->0, priority->1, spec_prior->2
        class_mapping = {
            'not_recom': 0,
            'priority': 1, 
            'spec_prior': 2
        }
        y_encoded = y_filtered.map(class_mapping)
        y_encoded = pd.Series(y_encoded, name='target').reset_index(drop=True)
        
        # Combine features and target
        X_encoded = X_encoded.reset_index(drop=True)
        df_final = pd.concat([X_encoded, y_encoded], axis=1)
        df_final = df_final.dropna()
        
        features = list(X_encoded.columns)
        classes = sorted(df_final['target'].unique())
        
        print(f"\nFinal dataset shape: {df_final.shape}")
        print(f"Features: {features}")
        print(f"Encoded classes: {classes}")
        print(f"Final class distribution:\n{df_final['target'].value_counts().sort_index()}")
        print(f"Missing values: {df_final.isnull().sum().sum()}")
        
        # Save for future use
        try:
            df_final.to_csv(data_file, index=False)
            print(f"Saved processed dataset to: {data_file}")
        except Exception as e:
            print(f"Warning: could not save dataset: {e}")
        
        return df_final, features, classes
        
    except ImportError:
        print("ERROR: ucimlrepo not found. Install with: pip install ucimlrepo")
        return None, None, None
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def create_logistic_regression_model():
    """Create Multinomial Logistic Regression model"""
    return LogisticRegression(max_iter=1000, multi_class='multinomial', random_state=42)


def create_xgboost_model():
    """Create XGBoost multi-class classifier"""
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost not available")
    return xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='multi:softprob',
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )


def create_lightgbm_model():
    """Create LightGBM multi-class classifier"""
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM not available")
    return lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='multiclass',
        random_state=42,
        verbose=-1
    )


def calculate_multiclass_brier_score(y_true, y_prob):
    """
    Calculate Brier score for multi-class classification
    Brier score = mean of squared differences between predicted probabilities and actual outcomes
    """
    n_classes = y_prob.shape[1]
    y_true_binary = label_binarize(y_true, classes=range(n_classes))
    brier_scores = []
    
    for i in range(n_classes):
        bs = brier_score_loss(y_true_binary[:, i], y_prob[:, i])
        brier_scores.append(bs)
    
    # Return average Brier score across all classes
    return np.mean(brier_scores)


def run_multi_imv_analysis(df_data, model_name, model_creator, seed, features):
    """
    Run Multi-IMV analysis for a given model and seed
    
    Returns:
        dict: Dictionary with IMV matrix and one-vs-all results
    """
    print(f"\n{'='*80}")
    print(f"Running {model_name} with seed {seed}")
    print(f"{'='*80}")
    
    # Create evaluator
    evaluator = MulticlassIMV(
        data=df_data,
        outcome_variable='target',
        optional_explanatory_variables=features,
        model_creator=model_creator,
        n_splits=10,
        random_state=seed
    )
    
    # Calculate IMV confusion matrix
    print("Calculating Multi-IMV confusion matrix...")
    imv_matrices_list, imv_matrix_avg = evaluator.k_fold_imv_matrix()
    
    # Calculate one-vs-all IMV
    print("Calculating one-vs-all IMV...")
    imv_ova_list, imv_ova_avg = evaluator.k_fold_one_vs_all()
    
    return {
        'imv_matrix_avg': imv_matrix_avg,
        'imv_matrices_list': imv_matrices_list,
        'imv_ova_avg': imv_ova_avg,
        'imv_ova_list': imv_ova_list
    }


def calculate_performance_metrics_multiclass(model_creator, df_data, features, seed, n_splits=10):
    """
    Calculate accuracy, precision, recall, and Brier score using k-fold cross-validation
    
    Returns:
        dict: Dictionary with all performance metrics
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    accuracies = []
    precisions = []
    recalls = []
    brier_scores = []
    
    X = df_data[features].values
    y = df_data['target'].values
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train model
        model = model_creator()
        model.fit(X_train, y_train)
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred) * 100
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0) * 100
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0) * 100
        brier = calculate_multiclass_brier_score(y_test, y_prob)
        
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        brier_scores.append(brier)
    
    return {
        'accuracy': accuracies,
        'precision': precisions,
        'recall': recalls,
        'brier': brier_scores
    }


def aggregate_imv_matrices(results_dict):
    """
    Aggregate IMV matrices across multiple seeds
    
    Args:
        results_dict: Dict of {seed: {'imv_matrix_avg': DataFrame}}
    
    Returns:
        tuple: (mean_matrix, std_matrix) as DataFrames
    """
    matrices = [results_dict[seed]['imv_matrix_avg'].values for seed in results_dict.keys()]
    matrices_stack = np.stack(matrices)
    
    mean_matrix = np.mean(matrices_stack, axis=0)
    std_matrix = np.std(matrices_stack, axis=0)
    
    # Get class labels from first matrix
    first_matrix = results_dict[list(results_dict.keys())[0]]['imv_matrix_avg']
    mean_df = pd.DataFrame(mean_matrix, index=first_matrix.index, columns=first_matrix.columns)
    std_df = pd.DataFrame(std_matrix, index=first_matrix.index, columns=first_matrix.columns)
    
    return mean_df, std_df


def save_model_results(model_name, imv_results, performance_results):
    """Save results for a specific model to disk"""
    results = {
        'imv_results': imv_results,
        'performance_results': performance_results
    }
    filename = os.path.join(RESULTS_DIR, f'nursery_{model_name.replace(" ", "_").lower()}_results.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    print(f"  ✓ Saved results to: {filename}")


def load_model_results(model_name):
    """Load results for a specific model from disk if they exist"""
    filename = os.path.join(RESULTS_DIR, f'nursery_{model_name.replace(" ", "_").lower()}_results.pkl')
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            results = pickle.load(f)
        print(f"  ✓ Loaded cached results from: {filename}")
        return results['imv_results'], results['performance_results']
    return None, None


def plot_imv_confusion_matrix(imv_matrix, model_name, class_names):
    """
    Create IMV confusion matrix heatmap (replicating Figure 3)
    
    Args:
        imv_matrix: DataFrame with IMV values
        model_name: Name of the model
        class_names: List of class names for labeling
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(imv_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Multi-IMV Value'},
                ax=ax, vmin=0, vmax=1)
    
    ax.set_title(f'Nursery ({model_name})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Class', fontsize=12)
    ax.set_ylabel('True Class', fontsize=12)
    
    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, f'nursery_multi_imv_{model_name.replace(" ", "_").lower()}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ IMV matrix plot saved to: {output_path}")
    plt.close()


def plot_all_imv_matrices(all_results, class_names, model_names_list):
    """
    Create side-by-side IMV confusion matrices for all models (Figure 3 style)
    """
    n_models = len(model_names_list)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (ax, model_name) in enumerate(zip(axes, model_names_list)):
        if model_name not in all_results:
            ax.set_title(f'{model_name}\n(Not Available)')
            ax.axis('off')
            continue
        
        mean_matrix, _ = all_results[model_name]
        
        # Create heatmap
        sns.heatmap(mean_matrix, annot=True, fmt='.3f', cmap='RdYlGn', 
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, vmin=0, vmax=1, cbar=False)
        
        ax.set_title(f'Nursery ({model_name})', fontsize=12, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('True Class', fontsize=11)
        ax.set_xlabel('Predicted Class', fontsize=11)
    
    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, 'nursery_multi_imv_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Combined IMV matrix plot saved to: {output_path}")
    plt.close()


def create_performance_table(performance_results, imv_results, class_names):
    """
    Create performance comparison table with Multi-IMV, Brier, Accuracy, Precision, Recall
    Replicates the second image format
    
    Args:
        performance_results: Dict of {model_name: {metric: [values]}}
        imv_results: Dict of {model_name: (mean_matrix, std_matrix)}
        class_names: List of class names
    """
    print("\n" + "="*120)
    print("PERFORMANCE COMPARISON TABLE (Nursery Dataset)")
    print("="*120)
    
    # Calculate average Multi-IMV for each class (one-vs-all style)
    # We'll use the diagonal and row averages as a proxy
    
    print(f"{'Methods':<25} ", end='')
    for class_name in class_names:
        print(f"{class_name:<20} ", end='')
    print()
    print("-"*120)
    
    model_order = ['Logistic Regression', 'XGBoost', 'LightGBM']
    
    for model_name in model_order:
        if model_name not in performance_results:
            continue
        
        perf = performance_results[model_name]
        
        # Multi-IMV values (average off-diagonal elements per class)
        if model_name in imv_results:
            mean_matrix, std_matrix = imv_results[model_name]
            n_classes = len(class_names)
            
            # For each class, calculate average IMV for discriminating it from others
            multi_imv_values = []
            for i in range(n_classes):
                # Average of row i (excluding diagonal)
                row_vals = [mean_matrix.iloc[i, j] for j in range(n_classes) if i != j]
                multi_imv_values.append((np.mean(row_vals), np.std(row_vals)))
        else:
            multi_imv_values = [(0, 0)] * len(class_names)
        
        # Print Multi-IMV
        print(f"Multi-IMV ({model_name[:3]})      ", end='')
        for mean_val, std_val in multi_imv_values:
            print(f"{mean_val:.2f}±{std_val:.2f}         ", end='')
        print()
        
        # Print Brier Loss
        brier_mean = np.mean(perf['brier'])
        brier_std = np.std(perf['brier'])
        print(f"Brier Loss ({model_name[:3]})     ", end='')
        for _ in class_names:
            print(f"{brier_mean:.2f}±{brier_std:.2f}         ", end='')
        print()
        
        # Print Accuracy
        acc_mean = np.mean(perf['accuracy'])
        acc_std = np.std(perf['accuracy'])
        print(f"Accuracy ({model_name[:3]})       ", end='')
        for _ in class_names:
            print(f"{acc_mean:.2f}±{acc_std:.2f}        ", end='')
        print()
        
        # Print Precision
        prec_mean = np.mean(perf['precision'])
        prec_std = np.std(perf['precision'])
        print(f"Precision ({model_name[:3]})      ", end='')
        for _ in class_names:
            print(f"{prec_mean:.2f}±{prec_std:.2f}        ", end='')
        print()
        
        # Print Recall
        rec_mean = np.mean(perf['recall'])
        rec_std = np.std(perf['recall'])
        print(f"Recall ({model_name[:3]})         ", end='')
        for _ in class_names:
            print(f"{rec_mean:.2f}±{rec_std:.2f}        ", end='')
        print()
        
    print("="*120)
    
    # Detailed metrics summary
    print("\nDETAILED METRICS SUMMARY")
    print("="*120)
    for model_name in model_order:
        if model_name not in performance_results:
            continue
        
        print(f"\n{model_name}:")
        perf = performance_results[model_name]
        
        print(f"  Accuracy:  {np.mean(perf['accuracy']):.2f} ± {np.std(perf['accuracy']):.2f}%")
        print(f"  Precision: {np.mean(perf['precision']):.2f} ± {np.std(perf['precision']):.2f}%")
        print(f"  Recall:    {np.mean(perf['recall']):.2f} ± {np.std(perf['recall']):.2f}%")
        print(f"  Brier:     {np.mean(perf['brier']):.4f} ± {np.std(perf['brier']):.4f}")
    
    print("\n" + "="*120)


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-IMV Nursery Analysis')
    parser.add_argument('--force', action='store_true', 
                       help='Force re-run all models (ignore cached results)')
    parser.add_argument('--clear-cache', action='store_true',
                       help='Clear all cached results and exit')
    args = parser.parse_args()
    
    # Clear cache if requested
    if args.clear_cache:
        import glob
        cache_files = glob.glob(os.path.join(RESULTS_DIR, 'nursery_*_results.pkl'))
        for f in cache_files:
            os.remove(f)
            print(f"Removed: {f}")
        print(f"✓ Cleared {len(cache_files)} cached result file(s)")
        return
    
    print("="*80)
    print("Multi-IMV Nursery Analysis")
    if args.force:
        print("(FORCE MODE: Ignoring cached results)")
    print("="*80)
    
    # Load and prepare data
    print("\n1. Loading and preparing Nursery dataset...")
    df_nursery, features, classes = load_and_prepare_nursery_data()
    
    if df_nursery is None or features is None:
        print("ERROR: Failed to load dataset. Exiting.")
        return
    
    # Class names for display
    class_names = ['Not recommend', 'Priority acceptance', 'Special priority']
    
    # Define models to test
    models = {
        'Logistic Regression': create_logistic_regression_model,
    }
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = create_xgboost_model
    
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = create_lightgbm_model
    
    # Run analysis with 3 different random seeds
    seeds = [42, 43, 44]
    
    all_imv_results = {}
    all_performance_results = {}
    
    for model_name, model_creator in models.items():
        print(f"\n{'='*80}")
        print(f"Analyzing {model_name}")
        print(f"{'='*80}")
        
        # Check if results already exist (unless force mode is enabled)
        if not args.force:
            cached_imv, cached_perf = load_model_results(model_name)
            if cached_imv is not None and cached_perf is not None:
                print(f"  Using cached results for {model_name}")
                mean_matrix, std_matrix = aggregate_imv_matrices(cached_imv)
                all_imv_results[model_name] = (mean_matrix, std_matrix)
                all_performance_results[model_name] = cached_perf
                continue
        else:
            print(f"  Force mode: Re-running analysis for {model_name}")
        
        model_imv_results = {}
        model_performance = {'accuracy': [], 'precision': [], 'recall': [], 'brier': []}
        
        for i, seed in enumerate(seeds, 1):
            print(f"\n  [{i}/{len(seeds)}] Running with seed {seed}...")
            
            # Run Multi-IMV analysis
            imv_result = run_multi_imv_analysis(
                df_nursery, model_name, model_creator, seed, features
            )
            model_imv_results[seed] = imv_result
            
            # Calculate performance metrics
            perf_metrics = calculate_performance_metrics_multiclass(
                model_creator, df_nursery, features, seed
            )
            
            model_performance['accuracy'].extend(perf_metrics['accuracy'])
            model_performance['precision'].extend(perf_metrics['precision'])
            model_performance['recall'].extend(perf_metrics['recall'])
            model_performance['brier'].extend(perf_metrics['brier'])
            
            print(f"  Seed {seed}: Accuracy={np.mean(perf_metrics['accuracy']):.2f}%, "
                  f"Brier={np.mean(perf_metrics['brier']):.4f}")
        
        # Save results immediately after completing this model
        save_model_results(model_name, model_imv_results, model_performance)
        
        # Aggregate IMV matrices across seeds
        mean_matrix, std_matrix = aggregate_imv_matrices(model_imv_results)
        all_imv_results[model_name] = (mean_matrix, std_matrix)
        all_performance_results[model_name] = model_performance
        
        # Plot individual model's IMV matrix
        plot_imv_confusion_matrix(mean_matrix, model_name, class_names)
    
    # Create visualizations and tables
    print("\n" + "="*80)
    print("3. Creating visualizations and tables...")
    print("="*80)
    
    plot_all_imv_matrices(all_imv_results, class_names, list(models.keys()))
    create_performance_table(all_performance_results, all_imv_results, class_names)
    
    print("\n" + "="*80)
    print("✓ Analysis complete!")
    print(f"✓ Results saved in: {RESULTS_DIR}")
    print(f"✓ Figures saved in: {FIGURES_DIR}")
    print("\nTip: Use --force to re-run all models, or --clear-cache to delete cached results")
    print("="*80)


if __name__ == '__main__':
    main()
