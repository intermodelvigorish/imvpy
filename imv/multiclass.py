"""
Multi-class IMV: Information Model Vigor for Multi-class Classification

This module extends the IMV framework to handle classification problems with
three or more classes. It provides two evaluation approaches:
1. One-vs-All IMV: Measures information gain for each class vs all others
2. Pairwise IMV Matrix: Creates a confusion-matrix-style IMV comparison between class pairs

All core IMV computations (ll, get_w) are imported from imv.core module,
eliminating code duplication across the package.

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns

# Import shared IMV core functions
from .core import ll, get_w


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
    """
    
    def __init__(self, data, outcome_variable, model_creator, n_splits=10, 
                 optional_explanatory_variables=None, random_state=None):
        self.data = data
        self.outcome_variable = outcome_variable
        self.model_creator = model_creator
        self.n_splits = n_splits
        self.optional_explanatory_variables = (
            optional_explanatory_variables 
            if optional_explanatory_variables is not None 
            else data.columns.drop(outcome_variable).tolist()
        )
        self.random_state = random_state

    # Note: ll() and get_w() are now imported from imv.core
    # No need to redefine them here - this eliminates code duplication!

    def multinominal_imv_matrix(self, data, outcome_variable, p_base, p_enhanced):
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
            
        Returns
        -------
        pd.DataFrame, shape (n_classes, n_classes)
            Pairwise IMV matrix with class labels as index/columns
            
        Process:
            1. For each class pair (i, j) where i ≠ j:
            2. Filter data to only samples of class i or j
            3. Normalize probabilities for binary comparison
            4. Compute IMV comparing base vs enhanced models
            5. Store IMV(i vs j) at position [i, j]
            
        Interpretation:
            - High IMV(i,j): Features help distinguish class i from class j
            - Low IMV(i,j): Little information gain for this class pair
            - Matrix is generally asymmetric: IMV(i,j) ≠ IMV(j,i)
        """
        outcomes = np.sort(data[outcome_variable].unique())
        imv_mat = np.zeros((len(outcomes), len(outcomes)))
        p_base = np.asarray(p_base)
        p_enhanced = np.asarray(p_enhanced)

        for i, outcome_i in enumerate(outcomes):
            for j, outcome_j in enumerate(outcomes):
                if i != j:
                    # Filter data to only include these two classes
                    mask = data[outcome_variable].isin([outcome_i, outcome_j])
                    data_tmp = data[mask]
                    y_tmp = np.where(data_tmp[outcome_variable] == outcome_i, 1, 0)
                    
                    # Normalize probabilities for binary comparison
                    p_b = p_base[mask, i] / np.sum(p_base[mask][:, [i, j]], axis=1)
                    p_e = p_enhanced[mask, i] / np.sum(p_enhanced[mask][:, [i, j]], axis=1)

                    # Use shared ll() and get_w() from core module
                    a0 = ll(y_tmp, p_b)
                    a1 = ll(y_tmp, p_e)

                    p0 = get_w(a0)
                    p1 = get_w(a1)
                    ew = (p1 - p0) / p0

                    imv_mat[i, j] = ew

        imv_mat = pd.DataFrame(imv_mat, index=outcomes, columns=outcomes)
        return imv_mat

    def one_vs_all_single_fold(self, data, outcome_variable, p_base, p_enhanced):
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
            
        Returns
        -------
        pd.DataFrame
            DataFrame with class labels and their IMV scores
        """
        p_base = np.asarray(p_base)
        p_enhanced = np.asarray(p_enhanced)

        outcomes = np.sort(data[outcome_variable].unique())
        imv = []

        # Determine if outcomes start at 0 or 1
        index_offset = 0 if outcomes[0] == 0 else 1

        for outcome in outcomes:
            # Binary encoding: current class vs all others
            y_tmp = np.where(data[outcome_variable] == outcome, 1, 0)

            p_b = p_base[:, outcome - index_offset]
            p_e = p_enhanced[:, outcome - index_offset]

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
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        imv_results = []

        for train_index, test_index in kf.split(self.data):
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
            ova_df = self.one_vs_all_single_fold(test_data, self.outcome_variable, p_base, p_enhanced)
            imv_results.append(ova_df['imv'].values)

        # Calculate average IMV across folds
        imv_average = np.mean(np.array(imv_results), axis=0)
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
            - Matrix is generally asymmetric
            - Uses random_state for reproducible splits
        """
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        imv_matrices_list = []

        for train_index, test_index in kf.split(self.data):
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
            imv_matrix = self.multinominal_imv_matrix(test_data, self.outcome_variable, p_base, p_enhanced)
            imv_matrices_list.append(imv_matrix.values)

        # Average across folds
        imv_matrices_average = np.mean(np.stack(imv_matrices_list), axis=0)
        imv_matrices_average = pd.DataFrame(
            imv_matrices_average, 
            index=imv_matrix.index, 
            columns=imv_matrix.columns
        )
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
