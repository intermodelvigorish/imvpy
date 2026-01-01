"""
SHAP-IMV Titanic Analysis
Compares Logistic Regression, XGBoost, and LightGBM using SHAP-IMV
Replicates results from Table 1 and Figure 2 of the paper
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from imv import BinaryIMV

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
DATA_DIR = os.path.join(os.path.dirname(EXAMPLE_DIR), 'data', 'titanic')
FIGURES_DIR = os.path.join(EXAMPLE_DIR, 'figures')
RESULTS_DIR = os.path.join(EXAMPLE_DIR, 'results')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_and_prepare_titanic_data():
    """
    Load and prepare the Titanic dataset with features as described in the paper:
    - Class (3-class: ticket class)
    - Sex (binary: gender)
    - Age (continuous)
    - Age*Class (interaction term)
    - Alone (binary: whether passenger was alone)
    - Fare (continuous: ticket fare)
    - Embarked (3-class: embarkation port)
    """
    train_file = os.path.join(DATA_DIR, 'train.csv')
    test_file = os.path.join(DATA_DIR, 'test.csv')
    
    # Load data
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    
    # Combine for preprocessing
    df_test['Survived'] = -1  # Placeholder
    df_combined = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    
    print(f"Combined dataset shape: {df_combined.shape}")
    print(f"Missing values:\n{df_combined.isnull().sum()}")
    
    # Feature engineering
    # 1. Class (already exists as Pclass)
    df_combined['Class'] = df_combined['Pclass']
    
    # 2. Sex (binary variable)
    df_combined['Sex'] = df_combined['Sex'].map({'male': 0, 'female': 1})
    
    # 3. Age (continuous) - mean imputation for missing values
    age_mean = df_combined['Age'].mean()
    df_combined['Age'] = df_combined['Age'].fillna(age_mean)
    
    # 4. Age*Class interaction term
    df_combined['Age*Class'] = df_combined['Age'] * df_combined['Class']
    
    # 5. Alone (binary: whether passenger was alone)
    # SibSp = siblings/spouses, Parch = parents/children
    df_combined['Alone'] = ((df_combined['SibSp'] + df_combined['Parch']) == 0).astype(int)
    
    # 6. Fare (continuous) - mean imputation for missing values
    fare_mean = df_combined['Fare'].mean()
    df_combined['Fare'] = df_combined['Fare'].fillna(fare_mean)
    
    # 7. Embarked (3-class: C, Q, S) - mode imputation and encode
    embarked_mode = df_combined['Embarked'].mode()[0]
    df_combined['Embarked'] = df_combined['Embarked'].fillna(embarked_mode)
    embarked_mapping = {'C': 0, 'Q': 1, 'S': 2}
    df_combined['Embarked'] = df_combined['Embarked'].map(embarked_mapping)
    
    # Filter to training data only (where Survived is known)
    df_final = df_combined[df_combined['Survived'] != -1].copy()
    
    # Select features for analysis
    features = ['Sex', 'Class', 'Age*Class', 'Age', 'Fare', 'Embarked', 'Alone']
    df_final = df_final[features + ['Survived']]
    df_final = df_final.rename(columns={'Survived': 'target'})
    
    print(f"\nFinal dataset shape: {df_final.shape}")
    print(f"Features: {features}")
    print(f"Missing values after preprocessing:\n{df_final.isnull().sum()}")
    
    return df_final, features


def create_logistic_regression_model():
    """Create Logistic Regression model"""
    return LogisticRegression(max_iter=1000, random_state=42)


def create_xgboost_model():
    """Create XGBoost model"""
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost not available")
    return xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )


def create_lightgbm_model():
    """Create LightGBM model"""
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM not available")
    return lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )


def run_shap_imv_analysis(df_combined, model_name, model_creator, seed, all_variables):
    """
    Run SHAP-IMV analysis for a given model and seed
    
    Returns:
        dict: Dictionary with SHAP-IMV values for each feature
    """
    print(f"\n{'='*80}")
    print(f"Running {model_name} with seed {seed}")
    print(f"{'='*80}")
    
    # Create evaluator
    evaluator = BinaryIMV(
        data=df_combined,
        outcome_variable='target',
        optional_explanatory_variables=all_variables,
        model_creator=model_creator,
        split_method='kfold',
        n_splits=10,
        prop_test=0.2,
        model_type='classification',
        random_seed=seed
    )
    
    # Run evaluation
    print("Running IMV evaluation...")
    evaluator.run_evaluation()
    
    # Calculate SHAP-IMV values
    print("Calculating SHAP-IMV values...")
    imvshapley_values = {}
    for variable in all_variables:
        imvshapley_values[variable] = evaluator.calculate_imvshapley_value(variable)
    
    return imvshapley_values, evaluator


def calculate_performance_metrics(model_creator, df_combined, all_variables, seed):
    """
    Calculate accuracy and precision using 10-fold cross-validation
    
    Returns:
        tuple: (accuracy, precision)
    """
    from sklearn.model_selection import cross_val_predict
    
    # Prepare data
    X = df_combined[all_variables]
    y = df_combined['target']
    
    # Get cross-validated predictions
    model = model_creator()
    y_pred = cross_val_predict(model, X, y, cv=10, n_jobs=-1)
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred) * 100
    precision = precision_score(y, y_pred, zero_division=0) * 100
    
    return accuracy, precision


def aggregate_shap_values(results_dict):
    """
    Aggregate SHAP-IMV values across multiple seeds
    
    Args:
        results_dict: Dict of {seed: {feature: shap_value}}
    
    Returns:
        tuple: (mean_values, std_values) as dictionaries
    """
    all_features = list(next(iter(results_dict.values())).keys())
    
    mean_values = {}
    std_values = {}
    
    for feature in all_features:
        values = [results_dict[seed][feature] for seed in results_dict.keys()]
        mean_values[feature] = np.mean(values)
        std_values[feature] = np.std(values)
    
    return mean_values, std_values


def save_model_results(model_name, shap_results, performance_results):
    """Save results for a specific model to disk"""
    results = {
        'shap_results': shap_results,
        'performance_results': performance_results
    }
    filename = os.path.join(RESULTS_DIR, f'titanic_{model_name.replace(" ", "_").lower()}_results.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    print(f"  ✓ Saved results to: {filename}")


def load_model_results(model_name):
    """Load results for a specific model from disk if they exist"""
    filename = os.path.join(RESULTS_DIR, f'titanic_{model_name.replace(" ", "_").lower()}_results.pkl')
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            results = pickle.load(f)
        print(f"  ✓ Loaded cached results from: {filename}")
        return results['shap_results'], results['performance_results']
    return None, None


def plot_shap_imv_comparison(results, model_names_list):
    """
    Create bar plots comparing SHAP-IMV values across models (replicating Figure 2)
    
    Args:
        results: Dict of {model_name: (mean_values, std_values)}
        model_names_list: List of model names in order
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    display_names = {
        'Logistic Regression': 'Titanic (Logistic Regression)',
        'XGBoost': 'Titanic (XGBoost)',
        'LightGBM': 'Titanic (LightGBM)'
    }
    
    for idx, (ax, model_name) in enumerate(zip(axes, model_names_list)):
        if model_name not in results:
            ax.set_title(f'{display_names[model_name]}\n(Not Available)')
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        mean_values, std_values = results[model_name]
        
        # Sort features by mean SHAP-IMV value (descending)
        sorted_features = sorted(mean_values.items(), key=lambda x: x[1], reverse=True)
        features = [f[0] for f in sorted_features]
        means = [f[1] for f in sorted_features]
        stds = [std_values[f[0]] for f in sorted_features]
        
        # Create bar plot
        x_pos = np.arange(len(features))
        bars = ax.bar(x_pos, means, yerr=stds, 
                      color=plt.cm.Greens(np.linspace(0.4, 0.8, len(features))),
                      capsize=5, alpha=0.8, edgecolor='black', linewidth=1.2)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(features, rotation=45, ha='right')
        ax.set_ylabel('Average SHAP-IMV Value', fontsize=11)
        ax.set_title(display_names[model_name], fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, max(means) * 1.2)
    
    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, 'titanic_shap_imv_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved to: {output_path}")
    plt.close()


def create_performance_table(performance_results, shap_results):
    """
    Create performance comparison table (replicating Table 1)
    
    Args:
        performance_results: Dict of {model_name: {'accuracy': [], 'precision': []}}
        shap_results: Dict of {model_name: (mean_values, std_values)}
    """
    print("\n" + "="*100)
    print("PERFORMANCE COMPARISON TABLE (Titanic Dataset)")
    print("="*100)
    
    # Table header
    print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Top 5 Features (by SHAP-IMV)'}")
    print("-"*100)
    
    for model_name in ['Logistic Regression', 'XGBoost', 'LightGBM']:
        if model_name not in performance_results:
            print(f"{model_name:<20} {'N/A':<12} {'N/A':<12} {'N/A'}")
            continue
        
        # Average performance metrics
        accuracies = performance_results[model_name]['accuracy']
        precisions = performance_results[model_name]['precision']
        
        avg_acc = np.mean(accuracies)
        avg_prec = np.mean(precisions)
        
        # Get top 5 features
        mean_values, _ = shap_results[model_name]
        top5_features = sorted(mean_values.items(), key=lambda x: x[1], reverse=True)[:5]
        top5_names = [f[0] for f in top5_features]
        
        print(f"{model_name:<20} {avg_acc:>6.2f}%      {avg_prec:>6.2f}%      {', '.join(top5_names)}")
    
    print("="*100)
    
    # Detailed SHAP-IMV values
    print("\nDETAILED SHAP-IMV VALUES (Mean ± Std)")
    print("="*100)
    
    for model_name in ['Logistic Regression', 'XGBoost', 'LightGBM']:
        if model_name not in shap_results:
            continue
        
        print(f"\n{model_name}:")
        mean_values, std_values = shap_results[model_name]
        sorted_features = sorted(mean_values.items(), key=lambda x: x[1], reverse=True)
        
        for feature, mean_val in sorted_features:
            std_val = std_values[feature]
            print(f"  {feature:<15} {mean_val:.4f} ± {std_val:.4f}")
    
    print("\n" + "="*100)


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SHAP-IMV Titanic Analysis')
    parser.add_argument('--force', action='store_true', 
                       help='Force re-run all models (ignore cached results)')
    parser.add_argument('--clear-cache', action='store_true',
                       help='Clear all cached results and exit')
    args = parser.parse_args()
    
    # Clear cache if requested
    if args.clear_cache:
        import glob
        cache_files = glob.glob(os.path.join(RESULTS_DIR, 'titanic_*_results.pkl'))
        for f in cache_files:
            os.remove(f)
            print(f"Removed: {f}")
        print(f"✓ Cleared {len(cache_files)} cached result file(s)")
        return
    
    print("="*80)
    print("SHAP-IMV Titanic Analysis")
    if args.force:
        print("(FORCE MODE: Ignoring cached results)")
    print("="*80)
    
    # Load and prepare data
    print("\n1. Loading and preparing Titanic dataset...")
    df_titanic, features = load_and_prepare_titanic_data()
    
    # Define models to test
    models = {
        'Logistic Regression': create_logistic_regression_model,
    }
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = create_xgboost_model
    
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = create_lightgbm_model
    
    # Run analysis with 3 different random seeds (as mentioned in paper)
    seeds = [42, 123, 456]
    
    all_results = {}
    performance_results = {}
    
    for model_name, model_creator in models.items():
        print(f"\n{'='*80}")
        print(f"Analyzing {model_name}")
        print(f"{'='*80}")
        
        # Check if results already exist (unless force mode is enabled)
        if not args.force:
            cached_shap, cached_perf = load_model_results(model_name)
            if cached_shap is not None and cached_perf is not None:
                print(f"  Using cached results for {model_name}")
                mean_values, std_values = aggregate_shap_values(cached_shap)
                all_results[model_name] = (mean_values, std_values)
                performance_results[model_name] = cached_perf
                continue
        else:
            print(f"  Force mode: Re-running analysis for {model_name}")
        
        model_shap_results = {}
        model_performance = {'accuracy': [], 'precision': []}
        
        for i, seed in enumerate(seeds, 1):
            print(f"\n  [{i}/{len(seeds)}] Running with seed {seed}...")
            
            # Run SHAP-IMV analysis
            shap_values, evaluator = run_shap_imv_analysis(
                df_titanic, model_name, model_creator, seed, features
            )
            model_shap_results[seed] = shap_values
            
            # Calculate performance metrics
            acc, prec = calculate_performance_metrics(
                model_creator, df_titanic, features, seed
            )
            model_performance['accuracy'].append(acc)
            model_performance['precision'].append(prec)
            
            print(f"  Seed {seed}: Accuracy={acc:.2f}%, Precision={prec:.2f}%")
        
        # Save results immediately after completing this model
        save_model_results(model_name, model_shap_results, model_performance)
        
        # Aggregate results across seeds
        mean_values, std_values = aggregate_shap_values(model_shap_results)
        all_results[model_name] = (mean_values, std_values)
        performance_results[model_name] = model_performance
    
    # Create visualizations and tables
    print("\n" + "="*80)
    print("3. Creating visualizations and tables...")
    print("="*80)
    
    plot_shap_imv_comparison(all_results, list(models.keys()))
    create_performance_table(performance_results, all_results)
    
    print("\n" + "="*80)
    print("✓ Analysis complete!")
    print(f"✓ Results saved in: {RESULTS_DIR}")
    print(f"✓ Figures saved in: {FIGURES_DIR}")
    print("\nTip: Use --force to re-run all models, or --clear-cache to delete cached results")
    print("="*80)


if __name__ == '__main__':
    main()
