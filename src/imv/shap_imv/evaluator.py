"""
SHAP-IMV: Shapley-based Information Model Vigor for Binary Classification

This module implements the SHAP-IMV (Shapley Additive Explanations - Information Model Vigor)
framework for evaluating feature importance in binary classification models.
It combines information theory and game theory to provide interpretable feature attributions.

All core IMV computations (ll, get_w, calculate_imv) are imported from imv.core module,
eliminating code duplication across the package.

"""

import warnings
from math import factorial
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Import shared IMV core functions
from ..utils.core import ll, get_w, calculate_imv

# Try to import tqdm_joblib, if not available, create a simple context manager
try:
    from tqdm_joblib import tqdm_joblib
except ImportError:
    import contextlib
    @contextlib.contextmanager
    def tqdm_joblib(tqdm_object, leave=False):
        """Fallback context manager if tqdm_joblib is not installed"""
        yield tqdm_object


class IncompleteCoalitionWarning(UserWarning):
    """The coalition results are missing subsets the exact Shapley sum requires.

    Emitted by :meth:`BinaryIMV.calculate_imvshapley_value`. Absent coalitions are
    substituted with IMV 0, so the result is not a valid Shapley value and will
    not satisfy additivity.
    """


class BinaryIMV:
    """
    Evaluator for computing Information Model Vigor (IMV) and SHAP-IMV values for binary classification.
    
    This class implements the IMV framework which measures the information gain
    from adding features to a model. It supports binary classification with
    k-fold cross-validation or train-test split evaluation strategies.
    
    Core IMV functions (ll, get_w, calculate_imv) are imported from imv.core module,
    ensuring consistency across all IMV implementations.
    
    Attributes:
        data (pd.DataFrame): Input dataset containing features and outcome
        outcome_variable (str): Name of the target/outcome column
        optional_explanatory_variables (list): List of feature column names to evaluate
        model_creator (callable): Function that returns a new model instance
        split_method (str): Evaluation method - 'kfold' or 'train_test_split'
        n_splits (int): Number of folds for k-fold cross-validation
        prop_test (float): Proportion of data for test set (train_test_split only)
        model_type (str): Must be 'classification'
        random_seed (int): Random seed for reproducibility
        all_combinations_imv (dict): Storage for computed IMV scores
        
    Examples:
        >>> from sklearn.linear_model import LogisticRegression
        >>> from imv.binary import BinaryIMV
        >>> 
        >>> evaluator = BinaryIMV(
        ...     data=df,
        ...     outcome_variable='target',
        ...     optional_explanatory_variables=['age', 'income', 'education'],
        ...     model_creator=lambda: LogisticRegression(max_iter=1000),
        ...     split_method='kfold',
        ...     n_splits=5,
        ...     prop_test=0.2,
        ...     model_type='classification'
        ... )
        >>> evaluator.run_evaluation()
        >>> evaluator.evaluate_imvshapley()
    """
    
    def __init__(self, data, outcome_variable, optional_explanatory_variables, model_creator, 
                 split_method='kfold', n_splits=5, prop_test=0.2,
                 model_type='classification', all_combinations_imv=None,
                 random_seed=42, n_jobs=1, verbose=False):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if outcome_variable not in data:
            raise ValueError(f"outcome variable {outcome_variable!r} is not in data")
        missing = set(optional_explanatory_variables) - set(data.columns)
        if missing:
            raise ValueError(f"unknown explanatory variables: {sorted(missing)}")
        if outcome_variable in optional_explanatory_variables:
            raise ValueError("outcome variable cannot also be an explanatory variable")
        valid_splits = {
            'kfold', 'stratified_kfold',
            'train_test_split', 'stratified_train_test_split',
        }
        if split_method not in valid_splits:
            raise ValueError(f"split_method must be one of {sorted(valid_splits)}")
        if model_type != 'classification':
            raise ValueError("only binary classification is methodologically supported")
        if data[outcome_variable].isna().any() or data[list(optional_explanatory_variables)].isna().any().any():
            raise ValueError("data contains missing values; impute or remove them first")
        if set(pd.unique(data[outcome_variable])) - {0, 1, False, True}:
            raise ValueError("binary classification outcome must contain only 0 and 1")
        self.data = data.copy()
        self.outcome_variable = outcome_variable
        self.optional_explanatory_variables = optional_explanatory_variables
        self.model_creator = model_creator
        self.split_method = split_method
        self.n_splits = n_splits
        self.prop_test = prop_test
        self.model_type = model_type
        self.random_seed = random_seed
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.data['constant'] = 1  # Add a constant column for modeling
        self.all_combinations_imv = all_combinations_imv if all_combinations_imv is not None else {}

    def calculate_imv_score(self, model_basic, model_enhanced, X_basic, X_enhanced, y):
        """
        Calculate IMV score comparing two trained models.
        
        Uses the shared calculate_imv() function from imv.core module.
        Computes IMV by comparing predictions from a basic model (intercept-only)
        and an enhanced model (with features). Handles both classification and
        binary classification tasks.
        
        Args:
            model_basic: Trained basic/null model (intercept only)
            model_enhanced: Trained enhanced model with features
            X_basic (pd.DataFrame): Basic features (constant only)
            X_enhanced (pd.DataFrame): Enhanced features (all variables)
            y (pd.Series): True labels/targets
            
        Returns:
            float: IMV score for this model comparison
            
        Raises:
            ValueError: If model_type is not 'classification'
            
        Note:
            Uses predict_proba()[:, 1] for positive class probability.
        """
        pred_basic = model_basic.predict_proba(X_basic)[:, 1]
        pred_enhanced = model_enhanced.predict_proba(X_enhanced)[:, 1]
        
        # Use shared IMV calculation from core module
        return calculate_imv(pred_basic, pred_enhanced, y)

    def compute_imv_method(self, combination):
        """
        Compute IMV for a specific combination of features.
        
        Trains models for a given feature subset and evaluates IMV score
        using either k-fold cross-validation or train-test split.
        
        Args:
            combination (tuple): Tuple of feature names to include in the model
            
        Returns:
            tuple: (combination, mean_imv_score, list_of_fold_scores)
                - combination: Input feature tuple
                - mean_imv_score: Average IMV across all folds/splits
                - list_of_fold_scores: Individual IMV score for each fold
                
        Note:
            - Always includes 'constant' column for intercept
            - Basic model uses only constant, enhanced model uses all features in combination
            - For kfold: Returns mean across all folds
            - For train_test_split: Returns single score in a list
        """
        X = self.data[list(combination) + ['constant']]
        y = self.data[self.outcome_variable]
        imv_scores = []
        fold_imv_scores = []

        if self.split_method in {'train_test_split', 'stratified_train_test_split'}:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.prop_test, random_state=self.random_seed,
                stratify=y if self.split_method == 'stratified_train_test_split' else None,
            )
            model_basic = self.model_creator()
            model_enhanced = self.model_creator()
            model_basic.fit(X_train[['constant']], y_train)
            model_enhanced.fit(X_train, y_train)
            imv_score = self.calculate_imv_score(model_basic, model_enhanced, X_test[['constant']], X_test, y_test)
            imv_scores.append(imv_score)
            fold_imv_scores.append(imv_score)
        elif self.split_method in {'kfold', 'stratified_kfold'}:
            if self.split_method == 'stratified_kfold':
                splitter = StratifiedKFold(
                    n_splits=self.n_splits, shuffle=True, random_state=self.random_seed
                )
                split_indices = splitter.split(X, y)
            else:
                splitter = KFold(
                    n_splits=self.n_splits, shuffle=True, random_state=self.random_seed
                )
                split_indices = splitter.split(X)
            for train_index, test_index in split_indices:
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                model_basic = self.model_creator()
                model_enhanced = self.model_creator()
                model_basic.fit(X_train[['constant']], y_train)
                model_enhanced.fit(X_train, y_train)
                imv_score = self.calculate_imv_score(model_basic, model_enhanced, X_test[['constant']], X_test, y_test)
                imv_scores.append(imv_score)
                fold_imv_scores.append(imv_score)
        return (tuple(combination), np.mean(imv_scores), fold_imv_scores)

    def run_evaluation(self):
        """
        Run IMV evaluation for all possible feature combinations.
        
        Computes IMV scores for all 2^n combinations of features (power set),
        where n is the number of optional explanatory variables. Uses parallel
        processing to speed up computation.
        
        Side Effects:
            - Populates self.all_combinations_imv with results
            - Prints the best performing feature combination
            
        Process:
            1. Generate all possible feature subsets (including empty set)
            2. Compute IMV for each subset in parallel using all CPU cores
            3. Store results as {combination: (mean_imv, fold_scores)}
            4. Identify and report best performing combination
            
        Note:
            - Computational complexity: O(2^n * k * m)
              where n=features, k=folds, m=training time per model
            - Uses joblib.Parallel with n_jobs=-1 for maximum parallelization
            - Progress shown via tqdm progress bar
            
        Example:
            >>> evaluator.run_evaluation()
            Evaluating IMV combinations: 100%|██████████| 8/8
            Best explanatory variables' combination: ('age', 'income'), with the highest IMV score: 0.234
        """
        combinations_list = [
            subset
            for L in range(0, len(self.optional_explanatory_variables) + 1)
            for subset in combinations(self.optional_explanatory_variables, L)
        ]

        results = {}

        # Use joblib's Parallel and delayed for parallel computation with tqdm
        progress = tqdm(
            desc="Evaluating IMV combinations", total=len(combinations_list),
            disable=not self.verbose,
        )
        with tqdm_joblib(progress, leave=False):
            parallel_results = Parallel(n_jobs=self.n_jobs)(
                delayed(self.compute_imv_method)(subset) for subset in combinations_list
            )

        for result in parallel_results:
            results[result[0]] = (result[1], result[2])

        self.all_combinations_imv = results
        best_combination, (max_imv, best_fold_scores) = max(self.all_combinations_imv.items(), key=lambda item: item[1][0])
        if self.verbose:
            print(f"Best explanatory variables' combination: {best_combination}, with the highest IMV score: {max_imv}")
        return self.all_combinations_imv

    def plot_single_var_combinations_layered_violin_centralized_zero(self, ax=None, figsize=(6, 4)):
        """
        Plot violin plot for single-variable IMV scores.
        
        Creates a horizontal violin plot showing the distribution of IMV scores
        across folds for each individual variable (models with only one feature).
        Useful for understanding individual variable performance and consistency.
        
        Args:
            ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, creates new figure.
            figsize (tuple, optional): Figure size if creating new figure. Default: (6, 4)
            
        Returns:
            tuple or matplotlib.axes.Axes or None:
                - If ax=None: Returns (fig, ax) tuple
                - If ax provided: Returns ax
                - If no single-variable combinations: Returns None
                
        Visualization Details:
            - X-axis: IMV scores
            - Y-axis: Variable names
            - Inner quartiles shown within violins
            - Color scheme: coolwarm palette
            
        Note:
            Must call run_evaluation() before plotting.
            Only shows variables used individually (combination length = 1).
            
        Example:
            >>> fig, ax = evaluator.plot_single_var_combinations_layered_violin_centralized_zero()
            >>> plt.show()
        """
        single_var_combinations = {
            combination: fold_scores 
            for combination, (imv_score, fold_scores) in self.all_combinations_imv.items() 
            if len(combination) == 1
        }
        
        if single_var_combinations:
            data = []
            for combination, scores in single_var_combinations.items():
                for score in scores:
                    data.append({'Variable': ', '.join(combination), 'IMV Score': score})
            df = pd.DataFrame(data)

            max_abs_value = df['IMV Score'].abs().max()
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
                created_ax = True
            else:
                created_ax = False
                
            # hue+legend=False rather than a bare palette: seaborn removes the
            # bare-palette form in 0.14.
            sns.violinplot(x='IMV Score', y='Variable', data=df, inner='quartile',
                          hue='Variable', palette='coolwarm', legend=False,
                          orient='h', ax=ax)
            ax.set_title('Single Variable Model vs Null Model')
            
            if created_ax:
                return fig, ax
            return ax
        else:
            print("No single variable combinations to plot.")
            return None

    def _missing_coalitions(self, variable):
        """Coalitions the exact Shapley sum for *variable* needs but which are absent.

        Every subset S of the remaining features contributes both v(S) and
        v(S | {variable}), so the full power set is required.
        """
        others = [f for f in self.optional_explanatory_variables if f != variable]
        order = {f: i for i, f in enumerate(self.optional_explanatory_variables)}
        missing = set()
        for size in range(len(others) + 1):
            for subset in combinations(others, size):
                without = tuple(sorted(subset, key=order.__getitem__))
                with_var = tuple(sorted(subset + (variable,), key=order.__getitem__))
                for key in (without, with_var):
                    if key not in self.all_combinations_imv:
                        missing.add(key)
        return missing

    @staticmethod
    def calculate_weight(s_size, n):
        """
        Calculate Shapley weight for a given coalition size.
        
        Computes the weight used in Shapley value calculation, which depends
        on the size of the feature subset (coalition) and total number of features.
        
        Args:
            s_size (int): Size of the current feature subset (number of features)
            n (int): Total number of features
            
        Returns:
            float: Shapley weight for this coalition size
            
        Mathematical Formula:
            weight = |S|! * (n - |S| - 1)! / n!
            where |S| is the subset size
            
        Note:
            This weight ensures fair attribution in cooperative game theory.
            All weights for a given feature sum to 1.0.
        """
        return factorial(s_size) * factorial(n - s_size - 1) / factorial(n)

    def calculate_imvshapley_value(self, variable):
        """
        Calculate SHAP-IMV value for a single variable.
        
        Computes the Shapley value using IMV as the characteristic function.
        This provides a fair attribution of information gain to each feature
        by considering all possible feature coalitions.
        
        Args:
            variable (str): Name of the variable to compute SHAP-IMV for
            
        Returns:
            float: SHAP-IMV value for the variable (rounded to 3 decimals)
            
        Mathematical Formula:
            SHAP-IMV(v) = Σ [weight(|S|, n) * (IMV(S ∪ {v}) - IMV(S))]
            where sum is over all subsets S not containing v
            
        Process:
            1. For each combination containing the variable
            2. Compute marginal contribution: IMV(with v) - IMV(without v)
            3. Weight by Shapley weight based on coalition size
            4. Sum all weighted contributions
            
        Interpretation:
            - Positive: Variable adds information on average
            - Negative: Variable reduces model information (rare)
            - Magnitude: Average marginal contribution across all contexts
            
        Note:
            Prints the computed value and returns it.
            Must call run_evaluation() first to populate all_combinations_imv.
        """
        if variable not in self.optional_explanatory_variables:
            raise ValueError(
                f"{variable!r} is not one of the evaluated explanatory variables"
            )
        if not self.all_combinations_imv:
            raise ValueError(
                "no coalition results available; call run_evaluation() first"
            )
        # The player set is the declared feature universe, not whatever happens to
        # be in all_combinations_imv. Inferring it from the results would silently
        # rescale every Shapley weight if the power set were ever truncated.
        n = len(self.optional_explanatory_variables)
        missing = self._missing_coalitions(variable)
        if missing:
            warnings.warn(
                f"SHAP-IMV for {variable!r} needs {1 << n} coalition results but "
                f"{len(missing)} are missing, for example "
                f"{sorted(missing, key=len)[:3]}. Absent coalitions are treated as "
                f"IMV 0, so the returned value is not a valid Shapley value and "
                f"will not satisfy additivity. Re-run run_evaluation() over the "
                f"complete power set.",
                IncompleteCoalitionWarning,
                stacklevel=2,
            )
        imvshapley_value = 0.0
        for combination, (imv_score, _) in self.all_combinations_imv.items():
            if variable in combination:
                subset = tuple(x for x in combination if x != variable)
                subset_imv = self.all_combinations_imv.get(subset, (0.0, []))[0]
                s_size = len(subset)
                weight = self.calculate_weight(s_size, n)
                marginal_contribution = imv_score - subset_imv
                imvshapley_value += weight * marginal_contribution
        imvshapley_value = round(imvshapley_value, 3)
        if self.verbose:
            print(f"SHAP-IMV value for variable {variable}: {imvshapley_value}")
        return imvshapley_value

    def evaluate_imvshapley(self, ax=None, figsize=(12, 4)):
        """
        Compute and visualize SHAP-IMV values for all variables.
        
        Calculates SHAP-IMV values for each feature and creates a horizontal
        bar plot showing feature importance ranked from highest to lowest.
        
        Args:
            ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, creates new figure.
            figsize (tuple, optional): Base figure size. Height auto-adjusts based on number
                of variables (minimum 0.5 inches per variable). Default: (12, 4)
                
        Returns:
            tuple or matplotlib.axes.Axes:
                - If ax=None: Returns (fig, ax) tuple
                - If ax provided: Returns ax
                
        Visualization Details:
            - Bars sorted by SHAP-IMV value (descending)
            - Color gradient from coolwarm_r colormap
            - Values displayed on bars
            - Whitegrid style for easy reading
            
        Process:
            1. Compute SHAP-IMV for each variable
            2. Sort variables by importance (descending)
            3. Create colored horizontal bar plot
            4. Annotate bars with numeric values
            
        Note:
            Must call run_evaluation() before this method.
            Prints individual SHAP-IMV values during computation.
            
        Example:
            >>> fig, ax = evaluator.evaluate_imvshapley(figsize=(14, 6))
            SHAP-IMV value for variable age: 0.145
            SHAP-IMV value for variable income: 0.112
            ...
            >>> plt.tight_layout()
            >>> plt.show()
        """
        imvshapley_values = {}
        for variable in self.optional_explanatory_variables:
            imvshapley_values[variable] = self.calculate_imvshapley_value(variable)

        sorted_keys = [k for k, v in sorted(imvshapley_values.items(), key=lambda item: item[1], reverse=True)]
        sorted_values = [round(imvshapley_values[k], 4) for k in sorted_keys]

        n_bars = len(sorted_keys)
        total_height_in = max(figsize[1], n_bars * 0.5)

        if ax is None:
            fig, ax = plt.subplots(figsize=(figsize[0], total_height_in))
            created_ax = True
        else:
            created_ax = False

        sns.set_style("whitegrid")

        norm = plt.Normalize(min(sorted_values), max(sorted_values))
        cmap = plt.get_cmap("coolwarm_r")
        colors = cmap(norm(sorted_values))

        # hue+legend=False rather than a bare palette: seaborn removes the
        # bare-palette form in 0.14.
        sns.barplot(x=sorted_values, y=sorted_keys, hue=sorted_keys,
                    palette=list(colors), legend=False, orient='h', ax=ax)

        for index, value in enumerate(sorted_values):
            ax.text(value, index, str(value), va='center')

        ax.set_xlabel('IMVShapley Value')
        ax.set_title('IMVShapley Values')

        if created_ax:
            return fig, ax

        return ax


# Backward compatibility alias
IMVEvaluator = BinaryIMV
