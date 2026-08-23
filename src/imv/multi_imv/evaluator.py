"""
Multi-class IMV: InterModel Vigorish for Multi-class Classification

This module extends the IMV framework to handle classification problems with
three or more classes. It provides two evaluation approaches:
1. One-vs-All IMV: Measures information gain for each class vs all others
2. Pairwise IMV Matrix: Creates a confusion-matrix-style IMV comparison between class pairs

All core IMV computations (ll, get_w) are imported from imv.core module,
eliminating code duplication across the package.

"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import KFold, StratifiedKFold

# Import shared IMV core functions
from ..utils.core import get_w, ll


def _nanmean(values, axis):
    """Mean that ignores NaN without warning when a whole slice is NaN."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(values, axis=axis)


class MulticlassIMV:
    """
    Multinomial IMV for multi-class classification problems.
    
    This class extends IMV to handle classification tasks with 3 or more classes,
    providing both one-vs-all IMV scores and pairwise IMV confusion matrices.
    
    Parameters
    ----------
    data : pd.DataFrame
        The dataset containing features and outcome variable
    outcome_variable : str
        Name of the outcome/target column
    model_creator : callable
        Function that returns a fresh instance of the model
    n_splits : int, default=10
        Number of folds for k-fold cross-validation
    optional_explanatory_variables : list, optional
        List of feature column names. If None, uses all columns except outcome
    random_state : int, optional
        Random seed for reproducibility
    stratified : bool, default=False
        False reproduces the original notebooks' shuffled ``KFold``. True selects
        ``StratifiedKFold``, which is the right choice for a new analysis on
        imbalanced classes and guarantees no fold omits a class. Switching
        changes the result and must be reported.
    verbose : bool, default=False
        Print per-fold progress and results.
    """

    # Notebook-era compatibility while retaining one canonical implementation.
    # These must stay below the docstring: a class body statement placed above a
    # string literal turns that literal into a no-op expression, leaving
    # ``MulticlassIMV.__doc__`` as None.
    ll = staticmethod(ll)
    get_w = staticmethod(get_w)

    def __init__(self, data, outcome_variable, model_creator, n_splits=10, 
                 optional_explanatory_variables=None, random_state=None,
                 stratified=False, verbose=False):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if outcome_variable not in data:
            raise ValueError(f"outcome variable {outcome_variable!r} is not in data")
        self.data = data.copy()
        self.outcome_variable = outcome_variable
        self.model_creator = model_creator
        self.n_splits = n_splits
        self.optional_explanatory_variables = (
            optional_explanatory_variables 
            if optional_explanatory_variables is not None 
            else data.columns.drop(outcome_variable).tolist()
        )
        self.random_state = random_state
        self.stratified = bool(stratified)
        self.verbose = bool(verbose)

    def _split_indices(self):
        """Return original KFold indices or explicit production stratification."""
        if self.stratified:
            splitter = StratifiedKFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.random_state
            )
            return splitter.split(self.data, self.data[self.outcome_variable])
        splitter = KFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        return splitter.split(self.data)

    # Note: ll() and get_w() are now imported from imv.core
    # No need to redefine them here - this eliminates code duplication!

    @staticmethod
    def _fold_classes(model_basic, model_enhanced):
        """Column order shared by both fitted models' predict_proba output."""
        basic = np.asarray(model_basic.classes_)
        enhanced = np.asarray(model_enhanced.classes_)
        if not np.array_equal(basic, enhanced):
            raise ValueError(
                "null and enhanced models disagree on class order; their "
                "probability columns cannot be compared"
            )
        return enhanced

    @staticmethod
    def _resolve_classes(data, outcome_variable, p_base, p_enhanced, classes):
        """Label each probability column, validating shapes.

        ``classes`` names the class each column of *p_base*/*p_enhanced* belongs
        to and must be given in column order (i.e. ``model.classes_``). When it
        is omitted the columns are assumed to follow the sorted classes present
        in *data*, which is only correct if this fold contains every class the
        model was trained on.
        """
        p_base = np.asarray(p_base)
        p_enhanced = np.asarray(p_enhanced)
        if p_base.ndim != 2 or p_enhanced.shape != p_base.shape:
            raise ValueError("probability arrays must have matching (samples, classes) shapes")
        if p_base.shape[0] != len(data):
            raise ValueError("probability rows must match the number of observations")
        if classes is None:
            classes = np.sort(data[outcome_variable].unique())
            if p_base.shape[1] != len(classes):
                raise ValueError(
                    f"probability arrays have {p_base.shape[1]} columns but this fold "
                    f"contains {len(classes)} classes; pass classes=model.classes_ so "
                    "columns can be matched to labels, or use stratified folds"
                )
        else:
            classes = np.asarray(classes)
            if p_base.shape[1] != len(classes):
                raise ValueError("classes must name exactly one label per probability column")
        return classes, p_base, p_enhanced

    def multinominal_imv_matrix(self, data, outcome_variable, p_base, p_enhanced, classes=None):
        """
        Calculate pairwise IMV confusion matrix for all class combinations.
        
        Creates a matrix showing information gain for each pair of classes.
        Element (i,j) represents IMV when discriminating class i from class j.
        Diagonal elements are zero (no discrimination within same class).
        
        Parameters
        ----------
        data : pd.DataFrame
            Test data containing outcome variable
        outcome_variable : str
            Name of outcome/target column
        p_base : array-like, shape (n_samples, n_classes)
            Predicted probabilities from null model (intercept only)
        p_enhanced : array-like, shape (n_samples, n_classes)
            Predicted probabilities from model with features
        classes : array-like, optional
            Label of each probability column, in column order (``model.classes_``).
            Required whenever *data* may not contain every class the model was
            trained on; otherwise columns are matched to the wrong labels.

        Returns
        -------
        pd.DataFrame, shape (n_classes, n_classes)
            Pairwise IMV matrix indexed by *classes*. Pairs for which this fold
            holds no samples of one or both classes are NaN.


        Process:
            1. For each class pair (i, j) where i ≠ j:
            2. Filter data to only samples of class i or j
            3. Normalize probabilities for binary comparison
            4. Compute IMV comparing base vs enhanced models
            5. Store IMV(i vs j) at position [i, j]
            
        Interpretation:
            - High IMV(i,j): Features help distinguish class i from class j
            - Low IMV(i,j): Little information gain for this class pair
            - Matrix is exactly symmetric: IMV(i,j) == IMV(j,i). Pairwise
              renormalization gives p_j = 1 - p_i, and swapping i and j also
              flips the label, so ll() is unchanged because it is invariant
              under (y, p) -> (1-y, 1-p). This is unlike the ablation matrix,
              where the two models have independent likelihoods.
        """
        outcomes, p_base, p_enhanced = self._resolve_classes(
            data, outcome_variable, p_base, p_enhanced, classes
        )
        imv_mat = np.zeros((len(outcomes), len(outcomes)))
        labels = data[outcome_variable].to_numpy()

        for i, outcome_i in enumerate(outcomes):
            for j, outcome_j in enumerate(outcomes):
                if i == j:
                    continue
                # Filter data to only include these two classes. Column i always
                # belongs to outcome_i because both are positions in *outcomes*.
                mask = (labels == outcome_i) | (labels == outcome_j)
                y_tmp = np.where(labels[mask] == outcome_i, 1, 0)
                if y_tmp.size == 0 or y_tmp.min() == y_tmp.max():
                    # One of the two classes is absent from this fold, so the
                    # pair carries no discrimination signal here.
                    imv_mat[i, j] = np.nan
                    continue

                # Normalize probabilities for binary comparison
                denom_b = np.sum(p_base[mask][:, [i, j]], axis=1)
                denom_e = np.sum(p_enhanced[mask][:, [i, j]], axis=1)
                if np.any(denom_b <= 0) or np.any(denom_e <= 0):
                    raise ValueError(
                        f"pair ({outcome_i!r}, {outcome_j!r}) has zero combined "
                        "probability mass; cannot renormalize"
                    )
                p_b = p_base[mask, i] / denom_b
                p_e = p_enhanced[mask, i] / denom_e

                # Use shared ll() and get_w() from core module
                a0 = ll(y_tmp, p_b)
                a1 = ll(y_tmp, p_e)

                p0 = get_w(a0)
                p1 = get_w(a1)
                ew = (p1 - p0) / p0

                imv_mat[i, j] = ew

        imv_mat = pd.DataFrame(imv_mat, index=outcomes, columns=outcomes)
        return imv_mat

    def one_vs_all_single_fold(self, data, outcome_variable, p_base, p_enhanced, classes=None):
        """
        Calculate one-vs-all IMV for a single fold.
        
        For each class, calculates IMV treating it as positive class vs all others.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data with outcome variable
        outcome_variable : str
            Name of outcome column
        p_base : array-like
            Predicted probabilities from base model
        p_enhanced : array-like
            Predicted probabilities from enhanced model
        classes : array-like, optional
            Label of each probability column, in column order (``model.classes_``).
            Required whenever *data* may not contain every class the model was
            trained on; otherwise columns are matched to the wrong labels.

        Returns
        -------
        pd.DataFrame
            DataFrame with class labels and their IMV scores. Classes absent
            from this fold score NaN, since one-vs-rest is unmeasurable without
            positives.
        """
        outcomes, p_base, p_enhanced = self._resolve_classes(
            data, outcome_variable, p_base, p_enhanced, classes
        )
        labels = data[outcome_variable].to_numpy()
        imv = []

        for class_index, outcome in enumerate(outcomes):
            # Binary encoding: current class vs all others. class_index indexes
            # both *outcomes* and the probability columns, so they stay aligned.
            y_tmp = np.where(labels == outcome, 1, 0)
            if y_tmp.max() == 0:
                imv.append(np.nan)
                continue

            p_b = p_base[:, class_index]
            p_e = p_enhanced[:, class_index]

            # Use shared ll() and get_w() from core module
            a0 = ll(y_tmp, p_b)
            a1 = ll(y_tmp, p_e)

            p0 = get_w(a0)
            p1 = get_w(a1)

            ew = (p1 - p0) / p0
            imv.append(ew)

        ova_df = pd.DataFrame({'class': outcomes, 'imv': imv})
        return ova_df

    def k_fold_one_vs_all(self):
        """
        Perform k-fold cross-validation for one-vs-all IMV evaluation.
        
        Trains null and enhanced models across k folds and computes IMV for each
        class treated as positive vs all other classes combined as negative.
        
        Returns
        -------
        tuple of (imv_results, imv_average)
            imv_results : list of numpy arrays
                IMV scores for each fold, shape (n_folds, n_classes)
            imv_average : numpy array
                Mean IMV scores across all folds, shape (n_classes,)
                
        Process:
            1. Split data into k folds
            2. For each fold:
               - Train null model (constant only) on train set
               - Train enhanced model (with features) on train set
               - Compute one-vs-all IMV on test set
            3. Average IMV scores across all folds
            
        Side Effects:
            Prints IMV results for all folds to console.
            
        Example Output:
            IMV results across folds: [[0.15, 0.23, 0.18], [0.14, 0.21, 0.19], ...]
            
        Note:
            Uses random_state for reproducible fold splits if specified.
        """
        imv_results = []

        for train_index, test_index in self._split_indices():
            X_train = self.data[self.optional_explanatory_variables].iloc[train_index]
            X_test = self.data[self.optional_explanatory_variables].iloc[test_index]
            y_train = self.data[self.outcome_variable].iloc[train_index]
            y_test = self.data[self.outcome_variable].iloc[test_index]

            # Base model (null model with only constant)
            X_train_constant = np.ones((X_train.shape[0], 1))
            X_test_constant = np.ones((X_test.shape[0], 1))
            model_basic = self.model_creator()
            model_basic.fit(X_train_constant, y_train)
            p_base = model_basic.predict_proba(X_test_constant)

            # Enhanced model with features
            model_enhanced = self.model_creator()
            model_enhanced.fit(X_train, y_train)
            p_enhanced = model_enhanced.predict_proba(X_test)

            # Calculate IMV for this fold
            test_data = X_test.copy()
            test_data[self.outcome_variable] = y_test
            ova_df = self.one_vs_all_single_fold(
                test_data, self.outcome_variable, p_base, p_enhanced,
                classes=self._fold_classes(model_basic, model_enhanced),
            )
            imv_results.append(ova_df['imv'].values)

        # Folds that lack a class contribute NaN for it rather than a wrong value.
        imv_average = _nanmean(np.array(imv_results), axis=0)
        if self.verbose:
            print(f"IMV results across folds: {imv_results}")
        
        return imv_results, imv_average

    def k_fold_imv_matrix(self):
        """
        Perform k-fold cross-validation for pairwise IMV confusion matrix.
        
        Trains models across k folds and computes pairwise IMV matrices,
        then averages to get stable estimates of class discrimination ability.
        
        Returns
        -------
        tuple of (imv_matrices_list, imv_matrices_average)
            imv_matrices_list : list of numpy arrays
                IMV confusion matrix for each fold, shape (n_folds, n_classes, n_classes)
            imv_matrices_average : pd.DataFrame
                Average IMV matrix across folds, shape (n_classes, n_classes)
                with class labels as index/columns
                
        Process:
            1. Split data into k folds
            2. For each fold:
               - Train null model (constant only) on train set
               - Train enhanced model (with features) on train set  
               - Compute pairwise IMV matrix on test set
            3. Average matrices element-wise across all folds
            
        Side Effects:
            Prints the averaged IMV matrix to console.
            
        Example Output:
            Average IMV Matrix:
                   0      1      2
            0  0.000  0.145  0.123
            1  0.132  0.000  0.098
            2  0.115  0.102  0.000
            
        Note:
            - Diagonal elements are always 0 (no self-discrimination)
            - Matrix is exactly symmetric (see multinominal_imv_matrix)
            - Uses random_state for reproducible splits
        """
        imv_matrices_list = []

        for train_index, test_index in self._split_indices():
            X_train = self.data[self.optional_explanatory_variables].iloc[train_index]
            X_test = self.data[self.optional_explanatory_variables].iloc[test_index]
            y_train = self.data[self.outcome_variable].iloc[train_index]
            y_test = self.data[self.outcome_variable].iloc[test_index]

            # Base model (null model with only constant)
            X_train_constant = np.ones((X_train.shape[0], 1))
            X_test_constant = np.ones((X_test.shape[0], 1))
            model_basic = self.model_creator()
            model_basic.fit(X_train_constant, y_train)
            p_base = model_basic.predict_proba(X_test_constant)

            # Enhanced model with features
            model_enhanced = self.model_creator()
            model_enhanced.fit(X_train, y_train)
            p_enhanced = model_enhanced.predict_proba(X_test)

            # Calculate IMV matrix for this fold
            test_data = self.data.iloc[test_index].copy()
            test_data[self.outcome_variable] = y_test
            imv_matrix = self.multinominal_imv_matrix(
                test_data, self.outcome_variable, p_base, p_enhanced,
                classes=self._fold_classes(model_basic, model_enhanced),
            )
            imv_matrices_list.append(imv_matrix.values)

        # Folds that lack a class contribute NaN for it rather than a wrong value.
        imv_matrices_average = _nanmean(np.stack(imv_matrices_list), axis=0)
        imv_matrices_average = pd.DataFrame(
            imv_matrices_average, 
            index=imv_matrix.index, 
            columns=imv_matrix.columns
        )
        if self.verbose:
            print("Average IMV Matrix:")
            print(imv_matrices_average)
        
        return imv_matrices_list, imv_matrices_average

    def multinomial_IMV_heatmap(self, imv_matrix, ax=None, figsize=(6, 6)):
        """
        Create heatmap visualization of pairwise IMV confusion matrix.
        
        Visualizes the IMV matrix as a colored heatmap with annotations.
        Useful for identifying which class pairs are most distinguishable.
        
        Parameters
        ----------
        imv_matrix : pd.DataFrame or array-like, shape (n_classes, n_classes)
            Pairwise IMV matrix to visualize (from k_fold_imv_matrix)
        ax : matplotlib.axes.Axes, optional
            Existing axis to plot on. If None, creates new figure.
        figsize : tuple, default=(6, 6)
            Figure size (width, height) if creating new figure
            
        Returns
        -------
        tuple or matplotlib.axes.Axes
            - If ax=None: Returns (fig, ax) tuple
            - If ax provided: Returns ax
            
        Visualization Details:
            - Color scheme: coolwarm (red=high IMV, blue=low IMV)
            - Annotations: IMV values displayed in cells (3 decimal places)
            - Labels: "Outcome1", "Outcome2", etc. for rows and columns
            - Diagonal: Always 0 (no self-discrimination)
            
        Example:
            >>> imv_matrices, imv_avg = evaluator.k_fold_imv_matrix()
            >>> fig, ax = evaluator.multinomial_IMV_heatmap(imv_avg)
            >>> plt.tight_layout()
            >>> plt.show()
        """
        data = np.array(imv_matrix)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            created_ax = True
        else:
            fig = ax.figure
            created_ax = False

        num_rows, num_cols = data.shape
        xlabels = [f'Outcome{i+1}' for i in range(num_cols)]
        ylabels = [f'Outcome{i+1}' for i in range(num_rows)]

        sns.heatmap(data, annot=True, ax=ax, cmap='coolwarm', fmt=".3f", 
                   xticklabels=xlabels, yticklabels=ylabels)
        ax.set_title('IMV Confusion Matrix')
        
        if created_ax:
            return fig, ax
        return ax

    def multinomial_IMV_boxplot(self, imv_results, figsize=(6, 6), ax=None):
        """
        Create boxplot visualization of one-vs-all IMV distribution across folds.
        
        Shows the distribution and variability of IMV scores for each class
        across k-fold cross-validation. Useful for assessing stability and
        comparing class-wise information gain.
        
        Parameters
        ----------
        imv_results : list of arrays, shape (n_folds, n_classes)
            One-vs-all IMV results from k_fold_one_vs_all()
        figsize : tuple, default=(6, 6)
            Figure size (width, height) if creating new figure
        ax : matplotlib.axes.Axes, optional
            Existing axis to plot on. If None, creates new figure.
            
        Returns
        -------
        tuple or matplotlib.axes.Axes
            - If ax=None: Returns (fig, ax) tuple
            - If ax provided: Returns ax
            
        Visualization Details:
            - One boxplot per class showing distribution across folds
            - Box: Interquartile range (IQR) Q1-Q3
            - Whiskers: Extend to 1.5*IQR or data extremes
            - Median line: Shown within each box
            - Labels: "Outcome1", "Outcome2", etc. for each class
            
        Example:
            >>> imv_results, imv_avg = evaluator.k_fold_one_vs_all()
            >>> fig, ax = evaluator.multinomial_IMV_boxplot(imv_results)
            >>> plt.tight_layout()
            >>> plt.show()
            
        Note:
            Narrow boxes indicate stable IMV across folds.
            Wide boxes suggest fold-dependent performance.
        """
        data_matrix = np.array(imv_results)
        num_columns = data_matrix.shape[1]
        labels = [f'Outcome{i+1}' for i in range(num_columns)]

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            created_ax = True
        else:
            created_ax = False

        ax.boxplot(data_matrix, labels=labels)
        ax.set_title('Multinomial IMV across Different Outcomes')
        ax.set_ylabel('IMV Value')
        
        if created_ax:
            return fig, ax
        return ax


# Backward compatibility: maintain old class name
MultinomialIMV = MulticlassIMV
