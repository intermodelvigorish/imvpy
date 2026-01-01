"""
Multi-IMV Dry Bean Analysis
Compares Logistic Regression, XGBoost, and LightGBM using Multi-IMV for multi-class classification
Replicates results from Figure 3 of the paper
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


def load_and_prepare_dry_bean_data():
    """
    Load and prepare the Dry Bean dataset
    7 classes: Seker, Barbunya, Bombay, Cali, Dermosan, Horoz, Sira
    Features: 16 continuous variables about bean structural characteristics
    """
    data_file = os.path.join(DATA_DIR, 'dry_bean_processed.csv')
    
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
        
        print("Fetching Dry Bean dataset...")
        # Dry Bean dataset ID: 602
        dry_bean = fetch_ucirepo(id=602)
        
        X = dry_bean.data.features
        y = dry_bean.data.targets
        
        print(f"Original dataset shape: {X.shape}")
        print(f"Original features: {list(X.columns)}")
        
        # Prepare target variable
        if isinstance(y, pd.DataFrame):
            target_col = y.columns[0]
            y_series = y[target_col]
        else:
            y_series = y
        
        print(f"Original class distribution:\n{y_series.value_counts()}")
        
        # Encode target variable
        # Map bean varieties to numeric codes
        class_mapping = {
            'SEKER': 0,
            'BARBUNYA': 1,
            'BOMBAY': 2,
            'CALI': 3,
            'DERMASON': 4,
            'HOROZ': 5,
            'SIRA': 6
        }
        y_encoded = y_series.map(class_mapping)
        y_encoded = pd.Series(y_encoded, name='target').reset_index(drop=True)
        
        # Features are already numeric, just reset index
        X_encoded = X.reset_index(drop=True)
        
        # Combine features and target
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
    """Calculate Brier score for multi-class classification"""
    n_classes = y_prob.shape[1]
    y_true_binary = label_binarize(y_true, classes=range(n_classes))
    brier_scores = []
    
    for i in range(n_classes):
        bs = brier_score_loss(y_true_binary[:, i], y_prob[:, i])
        brier_scores.append(bs)
    
    return np.mean(brier_scores)


def run_multi_imv_analysis(df_data, model_name, model_creator, seed, features):
    """Run Multi-IMV analysis for a given model and seed"""
    print(f"\n{'='*80}")
    print(f"Running {model_name} with seed {seed}")
    print(f"{'='*80}")
    
    evaluator = MulticlassIMV(
        data=df_data,
        outcome_variable='target',
        optional_explanatory_variables=features,
        model_creator=model_creator,
        n_splits=10,
        random_state=seed
    )
    
    print("Calculating Multi-IMV confusion matrix...")
    imv_matrices_list, imv_matrix_avg = evaluator.k_fold_imv_matrix()
    
    print("Calculating one-vs-all IMV...")
    imv_ova_list, imv_ova_avg = evaluator.k_fold_one_vs_all()
    
    return {
        'imv_matrix_avg': imv_matrix_avg,
        'imv_matrices_list': imv_matrices_list,
        'imv_ova_avg': imv_ova_avg,
        'imv_ova_list': imv_ova_list
    }


def calculate_performance_metrics_multiclass(model_creator, df_data, features, seed, n_splits=10):
    """Calculate accuracy, precision, recall, and Brier score using k-fold CV"""
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
        
        model = model_creator()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
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
    """Aggregate IMV matrices across multiple seeds"""
    matrices = [results_dict[seed]['imv_matrix_avg'].values for seed in results_dict.keys()]
    matrices_stack = np.stack(matrices)
    
    mean_matrix = np.mean(matrices_stack, axis=0)
    std_matrix = np.std(matrices_stack, axis=0)
    
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
    filename = os.path.join(RESULTS_DIR, f'dry_bean_{model_name.replace(" ", "_").lower()}_results.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    print(f"  ✓ Saved results to: {filename}")


def load_model_results(model_name):
    """Load results for a specific model from disk if they exist"""
    filename = os.path.join(RESULTS_DIR, f'dry_bean_{model_name.replace(" ", "_").lower()}_results.pkl')
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            results = pickle.load(f)
        print(f"  ✓ Loaded cached results from: {filename}")
        return results['imv_results'], results['performance_results']
    return None, None


def plot_imv_confusion_matrix(imv_matrix, model_name, class_names):
    """Create IMV confusion matrix heatmap"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(imv_matrix, annot=True, fmt='.3f', cmap='RdYlGn', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Multi-IMV Value'},
                ax=ax, vmin=0, vmax=1)
    
    ax.set_title(f'Dry Bean ({model_name})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Class', fontsize=12)
    ax.set_ylabel('True Class', fontsize=12)
    
    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, f'dry_bean_multi_imv_{model_name.replace(" ", "_").lower()}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ IMV matrix plot saved to: {output_path}")
    plt.close()


def create_performance_table(performance_results, imv_results, class_names):
    """Create performance comparison table"""
    print("\n" + "="*140)
    print("PERFORMANCE COMPARISON TABLE (Dry Bean Dataset)")
    print("="*140)
    
    print(f"{'Methods':<25} ", end='')
    for class_name in class_names:
        print(f"{class_name:<15} ", end='')
    print()
    print("-"*140)
    
    model_order = ['Logistic Regression', 'XGBoost', 'LightGBM']
    
    for model_name in model_order:
        if model_name not in performance_results:
            continue
        
        perf = performance_results[model_name]
        
        if model_name in imv_results:
            mean_matrix, std_matrix = imv_results[model_name]
            n_classes = len(class_names)
            
            multi_imv_values = []
            for i in range(n_classes):
                row_vals = [mean_matrix.iloc[i, j] for j in range(n_classes) if i != j]
                multi_imv_values.append((np.mean(row_vals), np.std(row_vals)))
        else:
            multi_imv_values = [(0, 0)] * len(class_names)
        
        print(f"Multi-IMV ({model_name[:3]})      ", end='')
        for mean_val, std_val in multi_imv_values:
            print(f"{mean_val:.2f}±{std_val:.2f}      ", end='')
        print()
        
        brier_mean = np.mean(perf['brier'])
        brier_std = np.std(perf['brier'])
        print(f"Brier Loss ({model_name[:3]})     ", end='')
        for _ in class_names:
            print(f"{brier_mean:.2f}±{brier_std:.2f}      ", end='')
        print()
        
        acc_mean = np.mean(perf['accuracy'])
        acc_std = np.std(perf['accuracy'])
        print(f"Accuracy ({model_name[:3]})       ", end='')
        for _ in class_names:
            print(f"{acc_mean:.2f}±{acc_std:.2f}     ", end='')
        print()
        
        prec_mean = np.mean(perf['precision'])
        prec_std = np.std(perf['precision'])
        print(f"Precision ({model_name[:3]})      ", end='')
        for _ in class_names:
            print(f"{prec_mean:.2f}±{prec_std:.2f}     ", end='')
        print()
        
        rec_mean = np.mean(perf['recall'])
        rec_std = np.std(perf['recall'])
        print(f"Recall ({model_name[:3]})         ", end='')
        for _ in class_names:
            print(f"{rec_mean:.2f}±{rec_std:.2f}     ", end='')
        print()
        
    print("="*140)


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-IMV Dry Bean Analysis')
    parser.add_argument('--force', action='store_true', 
                       help='Force re-run all models (ignore cached results)')
    parser.add_argument('--clear-cache', action='store_true',
                       help='Clear all cached results and exit')
    args = parser.parse_args()
    
    if args.clear_cache:
        import glob
        cache_files = glob.glob(os.path.join(RESULTS_DIR, 'dry_bean_*_results.pkl'))
        for f in cache_files:
            os.remove(f)
            print(f"Removed: {f}")
        print(f"✓ Cleared {len(cache_files)} cached result file(s)")
        return
    
    print("="*80)
    print("Multi-IMV Dry Bean Analysis")
    if args.force:
        print("(FORCE MODE: Ignoring cached results)")
    print("="*80)
    
    print("\n1. Loading and preparing Dry Bean dataset...")
    df_bean, features, classes = load_and_prepare_dry_bean_data()
    
    if df_bean is None or features is None:
        print("ERROR: Failed to load dataset. Exiting.")
        return
    
    class_names = ['Seker', 'Barbunya', 'Bombay', 'Cali', 'Dermosan', 'Horoz', 'Sira']
    
    models = {
        'Logistic Regression': create_logistic_regression_model,
    }
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = create_xgboost_model
    
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = create_lightgbm_model
    
    seeds = [42, 43, 44]
    
    all_imv_results = {}
    all_performance_results = {}
    
    for model_name, model_creator in models.items():
        print(f"\n{'='*80}")
        print(f"Analyzing {model_name}")
        print(f"{'='*80}")
        
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
            
            imv_result = run_multi_imv_analysis(
                df_bean, model_name, model_creator, seed, features
            )
            model_imv_results[seed] = imv_result
            
            perf_metrics = calculate_performance_metrics_multiclass(
                model_creator, df_bean, features, seed
            )
            
            model_performance['accuracy'].extend(perf_metrics['accuracy'])
            model_performance['precision'].extend(perf_metrics['precision'])
            model_performance['recall'].extend(perf_metrics['recall'])
            model_performance['brier'].extend(perf_metrics['brier'])
            
            print(f"  Seed {seed}: Accuracy={np.mean(perf_metrics['accuracy']):.2f}%, "
                  f"Brier={np.mean(perf_metrics['brier']):.4f}")
        
        save_model_results(model_name, model_imv_results, model_performance)
        
        mean_matrix, std_matrix = aggregate_imv_matrices(model_imv_results)
        all_imv_results[model_name] = (mean_matrix, std_matrix)
        all_performance_results[model_name] = model_performance
        
        plot_imv_confusion_matrix(mean_matrix, model_name, class_names)
    
    print("\n" + "="*80)
    print("3. Creating tables...")
    print("="*80)
    
    create_performance_table(all_performance_results, all_imv_results, class_names)
    
    print("\n" + "="*80)
    print("✓ Analysis complete!")
    print(f"✓ Results saved in: {RESULTS_DIR}")
    print(f"✓ Figures saved in: {FIGURES_DIR}")
    print("="*80)


if __name__ == '__main__':
    main()
