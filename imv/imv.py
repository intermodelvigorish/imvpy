"""
imv.py — Information Metric Value (IMV) evaluators for ML models.
Created by valler on 31/12/2024.

This module provides tools to:
- evaluate the incremental information value (IMV) of adding explanatory variables,
- search over all subsets of optional variables (binary classification),
- compute fold-level IMV scores, and
- attribute IMV to features via a Shapley-style decomposition ("IMVShapley").
- compute IMV for multiclass settings (one-vs-all and pairwise matrices).

Notes
-----
* The binary IMV formulation assumes classification with y in {0, 1} and models
  that output probabilities. The classic Bernoulli likelihood is used.
* A regression path is not implemented here; to support it, provide a proper
  likelihood (e.g., Gaussian/Poisson) and adapt the IMV math accordingly.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from itertools import combinations
from math import factorial
from concurrent.futures import ThreadPoolExecutor, as_completed

from scipy.optimize import minimize
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from tqdm import tqdm

# Plotting is optional; guard so the module can be imported without these libs.
try:  # pragma: no cover
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:  # pragma: no cover
    plt = None
    sns = None

# Shared math helpers (see utils.py)
# utils.py should define:
# - bernoulli_geomean_ll(y_true, p, eps=1e-12)
# - solve_w(a, guess=0.75, bounds=((1e-3, 0.9999),))
# - imv_from_probs(p_basic, p_enh, y_true)
from utils import bernoulli_geomean_ll, solve_w, imv_from_probs  # noqa: E402


# --------------------------------------------------------------------------- #
# Types
# --------------------------------------------------------------------------- #

ModelFactory = Callable[[], object]


# --------------------------------------------------------------------------- #
# Binary evaluator (subsets search + Shapley)
# --------------------------------------------------------------------------- #

class IMVEvaluator:
    """
    Evaluate IMV for all subsets of optional features (binary classification).

    Parameters
    ----------
    data : pd.DataFrame
        Input data containing outcome and candidate explanatory variables.
    outcome_variable : str
        Column name of the binary outcome (values in {0, 1}).
    optional_explanatory_variables : Sequence[str]
        Candidate variables to consider in combinations.
    model_creator : Callable[[], sklearn-like estimator]
        Zero-argument factory returning a *fresh* estimator with:
          - .fit(X, y)
          - .predict_proba(X) -> probabilities for the positive class at [:, 1]
    split_method : {"train_test_split", "kfold"}, default "train_test_split"
        Cross-validation strategy.
    n_splits : int, default 5
        Number of KFold splits (used if split_method == "kfold").
    prop_test : float, default 0.2
        Test proportion for train_test_split (0 < prop_test < 1).
    model_type : {"classification"}, default "classification"
        Type of problem. Only classification is supported here.
    all_combinations_imv : dict, optional
        Pre-populated results mapping combination -> (mean_imv, fold_scores).
    random_seed : int, default 42
        Random seed for shuffling/splits.

    Attributes
    ----------
    all_combinations_imv : Dict[Tuple[str, ...], Tuple[float, List[float]]]
        Computed IMV results per combination.
    best_combination_ : Optional[Tuple[str, ...]]
        Best combination found after run_evaluation().
    best_imv_ : Optional[float]
        Best mean IMV corresponding to best_combination_.
    best_fold_scores_ : Optional[List[float]]
        Fold-level IMV scores for the best combination.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        outcome_variable: str,
        optional_explanatory_variables: Sequence[str],
        model_creator: ModelFactory,
        split_method: str = "train_test_split",
        n_splits: int = 5,
        prop_test: float = 0.2,
        model_type: str = "classification",
        all_combinations_imv: Optional[Dict[Tuple[str, ...], Tuple[float, List[float]]]] = None,
        random_seed: int = 42,
    ) -> None:
        # Copy input data to prevent external mutation side effects
        self.data = data.copy()
        self.outcome_variable = outcome_variable
        self.optional_explanatory_variables = list(optional_explanatory_variables)
        self.model_creator = model_creator
        self.split_method = split_method
        self.n_splits = n_splits
        self.prop_test = prop_test
        self.model_type = model_type
        self.random_seed = random_seed

        # Add a uniquely named constant column (avoid collisions)
        const_name = "_imv_const"
        while const_name in self.data.columns:
            const_name = "_" + const_name
        self.const_col_ = const_name
        self.data[self.const_col_] = 1.0

        self.all_combinations_imv: Dict[Tuple[str, ...], Tuple[float, List[float]]] = (
            all_combinations_imv if all_combinations_imv is not None else {}
        )
        self.best_combination_: Optional[Tuple[str, ...]] = None
        self.best_imv_: Optional[float] = None
        self.best_fold_scores_: Optional[List[float]] = None

        # Validations
        if self.split_method not in {"train_test_split", "kfold"}:
            raise ValueError("split_method must be 'train_test_split' or 'kfold'.")
        if self.model_type != "classification":
            raise NotImplementedError(
                "This IMV uses Bernoulli likelihood (classification). "
                "Provide a regression likelihood to extend this to regression."
            )
        if not (0.0 < self.prop_test < 1.0):
            raise ValueError("prop_test must be in (0, 1).")

    # ------------------------------ Core IMV -------------------------------- #

    @staticmethod
    def _bernoulli_geometric_mean_likelihood(x: np.ndarray, p: np.ndarray) -> float:
        """
        Geometric mean of per-example Bernoulli likelihoods:
            exp( mean( x*log p + (1-x)*log (1-p) ) )
        """
        epsilon = 1e-12
        p = np.clip(p, epsilon, 1 - epsilon)
        z = x * np.log(p) + (1 - x) * np.log(1 - p)
        return float(np.exp(np.mean(z)))

    @staticmethod
    def _entropy_gap_objective(p: float, a: float) -> float:
        """| p log p + (1-p) log(1-p) - log(a) |."""
        return abs((p * np.log(p)) + ((1 - p) * np.log(1 - p)) - np.log(a))

    @classmethod
    def _solve_w(
        cls,
        a: float,
        guess: float = 0.75,
        bounds: Optional[Tuple[Tuple[float, float], ...]] = None,
    ) -> float:
        """
        Solve for w in (0.5, 1) such that:
            p log p + (1-p) log(1-p) ≈ log(a)
        """
        if bounds is None:
            bounds = ((0.5000001, 0.9999),)
        res = minimize(
            lambda p: cls._entropy_gap_objective(p, a),
            x0=np.array([guess]),
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1e-12, "gtol": 1e-10, "maxiter": 500},
        )
        w = float(res.x[0])
        if not (0.5 < w < 1.0):  # sanity/fallback
            w = 0.999
        return w

    @classmethod
    def calculate_imv(cls, y_basic: np.ndarray, y_enhanced: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute IMV = (w1 - w0) / w0 based on geometric-mean likelihoods.
        """
        ll_basic = bernoulli_geomean_ll(y_true, y_basic)
        ll_enhanced = bernoulli_geomean_ll(y_true, y_enhanced)
        w0 = solve_w(ll_basic)
        w1 = solve_w(ll_enhanced)
        if w0 <= 1e-12:
            return np.nan
        return float((w1 - w0) / w0)

    def _calculate_imv_score(
        self,
        model_basic,
        model_enhanced,
        X_basic: pd.DataFrame,
        X_enhanced: pd.DataFrame,
        y: pd.Series,
    ) -> float:
        """
        Calculate the IMV score for fitted models on a test split.
        """
        if not hasattr(model_basic, "predict_proba") or not hasattr(model_enhanced, "predict_proba"):
            raise AttributeError("Models must implement predict_proba(X) for classification IMV.")
        pred_basic = model_basic.predict_proba(X_basic)[:, 1]
        pred_enhanced = model_enhanced.predict_proba(X_enhanced)[:, 1]
        return imv_from_probs(pred_basic, pred_enhanced, y.to_numpy())

    # ----------------------------- Per subset -------------------------------- #

    def _evaluate_single_combination(
        self, combination: Tuple[str, ...]
    ) -> Tuple[Tuple[str, ...], float, List[float]]:
        """
        Fit/score a single combination of variables under the chosen split strategy.
        Returns (combination, mean_imv, fold_scores).
        """
        cols = list(combination) + [self.const_col_]
        X = self.data[cols]
        y = self.data[self.outcome_variable].astype(int).to_numpy()

        imv_scores: List[float] = []
        fold_imv_scores: List[float] = []

        if self.split_method == "train_test_split":
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.prop_test, random_state=self.random_seed, stratify=y
            )
            model_basic = self.model_creator()
            model_enhanced = self.model_creator()
            model_basic.fit(X_train[[self.const_col_]], y_train)
            model_enhanced.fit(X_train, y_train)

            score = self._calculate_imv_score(
                model_basic, model_enhanced, X_test[[self.const_col_]], X_test, pd.Series(y_test)
            )
            imv_scores.append(score)
            fold_imv_scores.append(score)

        elif self.split_method == "kfold":
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_seed)
            X_np = X.to_numpy()
            for train_index, test_index in kf.split(X_np):
                X_train, X_test = X_np[train_index], X_np[test_index]
                y_train, y_test = y[train_index], y[test_index]

                X_train_df = pd.DataFrame(X_train, columns=cols)
                X_test_df = pd.DataFrame(X_test, columns=cols)

                model_basic = self.model_creator()
                model_enhanced = self.model_creator()
                model_basic.fit(X_train_df[[self.const_col_]], y_train)
                model_enhanced.fit(X_train_df, y_train)

                score = self._calculate_imv_score(
                    model_basic, model_enhanced, X_test_df[[self.const_col_]], X_test_df, pd.Series(y_test)
                )
                imv_scores.append(score)
                fold_imv_scores.append(score)

        return (tuple(combination), float(np.nanmean(imv_scores)), fold_imv_scores)

    # ------------------------------ Grid search ------------------------------ #

    def run_evaluation(self, max_workers: Optional[int] = None) -> Dict[Tuple[str, ...], Tuple[float, List[float]]]:
        """
        Evaluate IMV over all subsets of the optional explanatory variables.

        Parameters
        ----------
        max_workers : Optional[int]
            Number of parallel workers. Uses ThreadPoolExecutor default if None.

        Returns
        -------
        Dict[Tuple[str, ...], Tuple[float, List[float]]]
            Mapping: combination -> (mean_imv, fold_scores)
        """
        combinations_list: List[Tuple[str, ...]] = [
            subset
            for L in range(0, len(self.optional_explanatory_variables) + 1)
            for subset in combinations(self.optional_explanatory_variables, L)
        ]

        results: Dict[Tuple[str, ...], Tuple[float, List[float]]] = {}
        
        # Use ThreadPoolExecutor for parallel evaluation
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._evaluate_single_combination, subset): subset
                for subset in combinations_list
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating IMV combinations"):
                combination = futures[future]
                try:
                    comb, mean_imv, fold_scores = future.result()
                    results[comb] = (mean_imv, fold_scores)
                except Exception as e:
                    print(f"Combination {combination} generated an exception: {e}")

        self.all_combinations_imv = results

        if results:
            self.best_combination_, (self.best_imv_, self.best_fold_scores_) = max(
                results.items(), key=lambda item: (item[1][0] if item[1][0] is not None else -np.inf)
            )
            print(
                f"Best explanatory variables' combination: {self.best_combination_}, "
                f"with the highest IMV score: {self.best_imv_}"
            )
        else:
            self.best_combination_ = None
            self.best_imv_ = None
            self.best_fold_scores_ = None
            print("No results computed.")

        return results

    # ------------------------------ Visualization --------------------------- #

    def plot_single_var_combinations_layered_violin_centralized_zero(
        self, ax=None, figsize: Tuple[float, float] = (6, 4)
    ):
        """
        Plot fold-level IMV distributions for single-variable models vs. null.
        """
        if plt is None or sns is None:
            raise ImportError("matplotlib and seaborn are required for plotting.")

        single_var_combinations = {
            combination: fold_scores
            for combination, (imv_score, fold_scores) in self.all_combinations_imv.items()
            if len(combination) == 1 and len(fold_scores) > 0
        }
        if not single_var_combinations:
            print("No single variable combinations to plot.")
            return

        data_rows: List[Dict[str, float]] = []
        for combination, scores in single_var_combinations.items():
            for score in scores:
                data_rows.append({"Variable": ", ".join(combination), "IMV Score": score})
        df = pd.DataFrame(data_rows)

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        sns.violinplot(x="IMV Score", y="Variable", data=df, inner="quartile", orient="h", ax=ax)
        ax.axvline(0.0, linewidth=1, linestyle="--")
        ax.set_title("Single Variable Model vs Null Model")
        ax.set_xlabel("IMV Score")
        ax.set_ylabel("Variable")
        return ax

    # ------------------------------ IMVShapley ------------------------------ #

    @staticmethod
    def _shapley_weight(s_size: int, n: int) -> float:
        """Weight: s! * (n - s - 1)! / n!"""
        return factorial(s_size) * factorial(n - s_size - 1) / factorial(n)

    def calculate_imvshapley_value(self, variable: str) -> float:
        """
        Compute IMVShapley value for a given variable using all_combinations_imv.
        """
        if not self.all_combinations_imv:
            raise RuntimeError("Run `run_evaluation()` before computing IMVShapley.")

        # Use the maximum observed coalition size or fallback to the total optional count
        n = max((len(comb) for comb in self.all_combinations_imv.keys()), default=0)
        n = max(n, len(self.optional_explanatory_variables))

        imvshapley_value = 0.0
        for combination, (imv_score, _) in self.all_combinations_imv.items():
            if variable in combination:
                subset = tuple(x for x in combination if x != variable)
                subset_imv = self.all_combinations_imv.get(subset, (0.0, []))[0]
                s_size = len(subset)
                weight = self._shapley_weight(s_size, n)
                marginal_contribution = (imv_score or 0.0) - (subset_imv or 0.0)
                imvshapley_value += weight * marginal_contribution

        imvshapley_value = round(float(imvshapley_value), 3)
        print(f"SHAP-IMV value for variable {variable}: {imvshapley_value}")
        return imvshapley_value

    def evaluate_imvshapley(self, ax=None, figsize: Tuple[float, float] = (12, 4)):
        """
        Compute and (optionally) plot IMVShapley values for all optional variables.
        """
        if not self.all_combinations_imv:
            raise RuntimeError("Run `run_evaluation()` before IMVShapley.")

        values: Dict[str, float] = {
            var: self.calculate_imvshapley_value(var) for var in self.optional_explanatory_variables
        }

        # If plotting libs are not available, return the dict
        if plt is None or sns is None:
            return values

        sorted_items = sorted(values.items(), key=lambda item: item[1], reverse=True)
        keys = [k for k, _ in sorted_items]
        vals = [round(v, 4) for _, v in sorted_items]

        total_height_in = max(figsize[1], len(keys) * 0.5)
        if ax is None:
            _, ax = plt.subplots(figsize=(figsize[0], total_height_in))
            created_ax = True
        else:
            created_ax = False

        sns.set_style("whitegrid")

        vmin, vmax = (min(vals), max(vals))
        if np.isclose(vmin, vmax):
            palette = ["C0"] * len(vals)
        else:
            norm = plt.Normalize(vmin, vmax)
            cmap = plt.get_cmap("coolwarm_r")
            palette = cmap(norm(vals))

        sns.barplot(x=vals, y=keys, palette=palette, orient="h", ax=ax)

        for idx, value in enumerate(vals):
            ax.text(value, idx, str(value), va="center")

        ax.set_xlabel("IMVShapley Value")
        ax.set_title("IMVShapley Values")

        if created_ax:
            return ax.figure, ax
        return ax


# --------------------------------------------------------------------------- #
# Multiclass evaluator (one-vs-all + pairwise)
# --------------------------------------------------------------------------- #

class MultinomialIMV:
    """
    Multinomial IMV evaluator (multiclass).

    Two views:
      1) One-vs-All IMV per class across folds.
      2) Pairwise IMV matrix comparing class i vs class j (i != j).

    Parameters
    ----------
    data : pd.DataFrame
        Dataset with explanatory features and the outcome column.
    outcome_variable : str
        Name of the multiclass target column.
    model_creator : Callable[[], estimator]
        Factory returning a classifier with .fit(X, y), .predict_proba(X),
        and a `classes_` attribute after fit.
    n_splits : int, default 10
        Number of folds (StratifiedKFold).
    optional_explanatory_variables : Sequence[str], optional
        Features to use; default is all columns except the outcome.
    random_state : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        outcome_variable: str,
        model_creator: ModelFactory,
        n_splits: int = 10,
        optional_explanatory_variables: Optional[Sequence[str]] = None,
        random_state: Optional[int] = None,
    ):
        self.data = data.copy()
        self.outcome_variable = outcome_variable
        self.model_creator = model_creator
        self.n_splits = n_splits
        self.random_state = random_state

        if optional_explanatory_variables is None:
            self.optional_explanatory_variables = [
                c for c in self.data.columns if c != self.outcome_variable
            ]
        else:
            self.optional_explanatory_variables = list(optional_explanatory_variables)

        # Global class order for alignment across folds/models
        self.outcomes_ = np.array(sorted(self.data[self.outcome_variable].unique()))

    # --------------------- Probability alignment helper --------------------- #

    @staticmethod
    def _align_proba_columns(
        proba: np.ndarray, model_classes: np.ndarray, desired_order: np.ndarray
    ) -> np.ndarray:
        """
        Align probability columns to a global `desired_order` of class labels.
        Missing classes get zero-prob columns; rows are renormalized to sum to 1.
        """
        cls_to_idx = {c: i for i, c in enumerate(model_classes)}
        n_samples = proba.shape[0]
        aligned = np.zeros((n_samples, len(desired_order)), dtype=float)
        for j, lbl in enumerate(desired_order):
            idx = cls_to_idx.get(lbl, None)
            aligned[:, j] = proba[:, idx] if idx is not None else 0.0
        row_sums = aligned.sum(axis=1, keepdims=True)
        mask = row_sums.squeeze() > 0
        aligned[mask] = aligned[mask] / row_sums[mask]
        return aligned

    @staticmethod
    def _binary_imv(y_binary: np.ndarray, p_basic: np.ndarray, p_enh: np.ndarray) -> float:
        """IMV for a binary reduced task."""
        return imv_from_probs(p_basic, p_enh, y_binary)

    # ---------------------------- Pairwise matrix --------------------------- #

    def multinomial_imv_matrix(
        self,
        data: pd.DataFrame,
        outcome_variable: str,
        p_base: np.ndarray,
        p_enhanced: np.ndarray,
        class_order: np.ndarray,
    ) -> pd.DataFrame:
        """
        Compute pairwise IMV(i, j) by reducing to binary tasks (i vs j).
        """
        outcomes = np.array(class_order)
        imv_mat = np.zeros((len(outcomes), len(outcomes)), dtype=float)

        for i, lbl_i in enumerate(outcomes):
            for j, lbl_j in enumerate(outcomes):
                if i == j:
                    continue

                mask = data[outcome_variable].isin([lbl_i, lbl_j]).to_numpy()
                if not np.any(mask):
                    continue

                y_bin = (data[outcome_variable].to_numpy()[mask] == lbl_i).astype(int)

                denom_b = p_base[mask][:, [i, j]].sum(axis=1)
                denom_e = p_enhanced[mask][:, [i, j]].sum(axis=1)
                denom_b = np.clip(denom_b, 1e-12, None)
                denom_e = np.clip(denom_e, 1e-12, None)

                p_b = p_base[mask, i] / denom_b
                p_e = p_enhanced[mask, i] / denom_e

                imv_mat[i, j] = self._binary_imv(y_bin, p_b, p_e)

        return pd.DataFrame(imv_mat, index=outcomes, columns=outcomes)

    # ----------------------------- One-vs-All ------------------------------- #

    def one_vs_all_single_fold(
        self,
        test_df: pd.DataFrame,
        outcome_variable: str,
        p_base: np.ndarray,
        p_enhanced: np.ndarray,
        class_order: np.ndarray,
    ) -> pd.DataFrame:
        """
        Compute OVA IMV per class on a single test split.
        """
        outcomes = np.array(class_order)
        y_test = test_df[outcome_variable].to_numpy()

        imv_per_class: List[float] = []
        for k, lbl in enumerate(outcomes):
            y_bin = (y_test == lbl).astype(int)
            imv_k = self._binary_imv(y_bin, p_base[:, k], p_enhanced[:, k])
            imv_per_class.append(imv_k)

        return pd.DataFrame({"class": outcomes, "imv": imv_per_class})

    # --------------------------- Cross-validation --------------------------- #

    def k_fold_one_vs_all(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Run Stratified K-Fold and compute OVA IMV per class on each fold.

        Returns
        -------
        list_of_folds : list of np.ndarray, each shape (n_classes,)
        average       : np.ndarray, shape (n_classes,)
        """
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        imv_results: List[np.ndarray] = []

        X_all = self.data[self.optional_explanatory_variables].to_numpy()
        y_all = self.data[self.outcome_variable].to_numpy()

        for train_idx, test_idx in skf.split(X_all, y_all):
            X_train, X_test = X_all[train_idx], X_all[test_idx]
            y_train, y_test = y_all[train_idx], y_all[test_idx]

            # Baseline (constant-only) model
            X_train_basic = np.ones((X_train.shape[0], 1))
            X_test_basic = np.ones((X_test.shape[0], 1))
            m_base = self.model_creator()
            m_base.fit(X_train_basic, y_train)
            p_base_raw = m_base.predict_proba(X_test_basic)
            classes_base = getattr(m_base, "classes_", None)
            if classes_base is None:
                raise AttributeError("Base model must expose `classes_` after fit.")

            # Enhanced model (features)
            m_enh = self.model_creator()
            m_enh.fit(X_train, y_train)
            p_enh_raw = m_enh.predict_proba(X_test)
            classes_enh = getattr(m_enh, "classes_", None)
            if classes_enh is None:
                raise AttributeError("Enhanced model must expose `classes_` after fit.")

            # Align both to the global outcomes_
            p_base = self._align_proba_columns(p_base_raw, classes_base, self.outcomes_)
            p_enhanced = self._align_proba_columns(p_enh_raw, classes_enh, self.outcomes_)

            # Build a test DataFrame with labels
            test_df = pd.DataFrame(X_test, columns=self.optional_explanatory_variables).assign(
                **{self.outcome_variable: y_test}
            )

            ova_df = self.one_vs_all_single_fold(
                test_df, self.outcome_variable, p_base, p_enhanced, self.outcomes_
            )
            imv_results.append(ova_df["imv"].to_numpy())

        imv_average = np.mean(np.vstack(imv_results), axis=0)
        return imv_results, imv_average

    def k_fold_imv_matrix(self) -> Tuple[List[np.ndarray], pd.DataFrame]:
        """
        Run Stratified K-Fold and compute the pairwise IMV matrix on each fold.

        Returns
        -------
        list_of_matrices : list of np.ndarray, each (n_classes, n_classes)
        average_matrix_df: pd.DataFrame (n_classes x n_classes)
        """
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        imv_matrices_list: List[np.ndarray] = []

        X_all = self.data[self.optional_explanatory_variables].to_numpy()
        y_all = self.data[self.outcome_variable].to_numpy()

        for train_idx, test_idx in skf.split(X_all, y_all):
            X_train, X_test = X_all[train_idx], X_all[test_idx]
            y_train, y_test = y_all[train_idx], y_all[test_idx]

            # Baseline (constant-only) model
            X_train_basic = np.ones((X_train.shape[0], 1))
            X_test_basic = np.ones((X_test.shape[0], 1))
            m_base = self.model_creator()
            m_base.fit(X_train_basic, y_train)
            p_base_raw = m_base.predict_proba(X_test_basic)
            classes_base = getattr(m_base, "classes_", None)
            if classes_base is None:
                raise AttributeError("Base model must expose `classes_` after fit.")

            # Enhanced model
            m_enh = self.model_creator()
            m_enh.fit(X_train, y_train)
            p_enh_raw = m_enh.predict_proba(X_test)
            classes_enh = getattr(m_enh, "classes_", None)
            if classes_enh is None:
                raise AttributeError("Enhanced model must expose `classes_` after fit.")

            p_base = self._align_proba_columns(p_base_raw, classes_base, self.outcomes_)
            p_enhanced = self._align_proba_columns(p_enh_raw, classes_enh, self.outcomes_)

            test_df = pd.DataFrame(X_test, columns=self.optional_explanatory_variables).assign(
                **{self.outcome_variable: y_test}
            )
            imv_df = self.multinomial_imv_matrix(
                data=test_df,
                outcome_variable=self.outcome_variable,
                p_base=p_base,
                p_enhanced=p_enhanced,
                class_order=self.outcomes_,
            )
            imv_matrices_list.append(imv_df.to_numpy())

        avg_matrix = np.mean(np.stack(imv_matrices_list), axis=0)
        avg_df = pd.DataFrame(avg_matrix, index=self.outcomes_, columns=self.outcomes_)
        return imv_matrices_list, avg_df

    # ------------------------------- Plotting -------------------------------- #

    def multinomial_imv_heatmap(self, imv_matrix: pd.DataFrame, ax=None, figsize: Tuple[float, float] = (6, 6)):
        """
        Visualize a pairwise IMV matrix as a heatmap.
        """
        if plt is None or sns is None:
            raise ImportError("matplotlib and seaborn are required for plotting.")

        data = np.asarray(imv_matrix)
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            data,
            annot=True,
            ax=ax,
            cmap="coolwarm",
            fmt=".3f",
            xticklabels=list(imv_matrix.columns),
            yticklabels=list(imv_matrix.index),
        )
        ax.set_title("IMV Pairwise Matrix")
        ax.set_xlabel("Class j")
        ax.set_ylabel("Class i")
        return ax

    def multinomial_imv_boxplot(self, imv_results: List[np.ndarray], ax=None, figsize: Tuple[float, float] = (6, 6)):
        """
        Visualize OVA IMV scores across folds as a boxplot per class.
        """
        if plt is None or sns is None:
            raise ImportError("matplotlib and seaborn are required for plotting.")

        data_matrix = np.vstack(imv_results)  # shape: (n_folds, n_classes)
        labels = [str(lbl) for lbl in self.outcomes_]

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        ax.boxplot(data_matrix, labels=labels)
        ax.set_title("Multinomial IMV across Outcomes (One-vs-All)")
        ax.set_xlabel("Class")
        ax.set_ylabel("IMV")
        return ax


# --------------------------------------------------------------------------- #
# Example usage
# --------------------------------------------------------------------------- #
# from sklearn.linear_model import LogisticRegression
#
# model_factory = lambda: LogisticRegression(max_iter=1000)
#
# # Binary evaluator
# evaluator = IMVEvaluator(
#     data=df,
#     outcome_variable="y",
#     optional_explanatory_variables=["x1", "x2", "x3"],
#     model_creator=model_factory,
#     split_method="kfold",
#     n_splits=5,
#     prop_test=0.2,
#     model_type="classification",
# )
# evaluator.run_evaluation()
# evaluator.evaluate_imvshapley()
#
# # Multiclass evaluator
# multi = MultinomialIMV(
#     data=df_multiclass,
#     outcome_variable="y_multi",
#     model_creator=model_factory,
#     n_splits=5,
# )
# imv_folds, imv_avg = multi.k_fold_one_vs_all()
# mats, mat_avg = multi.k_fold_imv_matrix()
# ax = multi.multinomial_imv_heatmap(mat_avg)
# --------------------------------------------------------------------------- #