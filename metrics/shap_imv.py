import numpy as np
from itertools import combinations
from tqdm import tqdm
from utils.core_imv import calculate_r2, calculate_ll, get_w, get_ew, get_vw

class IMVEvaluator:
    def __init__(self, data, outcome_variable, optional_explanatory_variables, model_creator, split_method, n_splits, prop_test, model_type, all_combinations_imv=None, random_seed=42):
        self.data = data
        self.outcome_variable = outcome_variable
        self.optional_explanatory_variables = optional_explanatory_variables
        self.model_creator = model_creator
        self.split_method = split_method
        self.n_splits = n_splits
        self.prop_test = prop_test
        self.model_type = model_type
        self.random_seed = random_seed  # Set the random seed
        self.data['constant'] = 1  # Add a constant column for modeling
        self.all_combinations_imv = all_combinations_imv if all_combinations_imv is not None else {}

    @staticmethod
    def ll(x, p):
        epsilon = 1e-4
        z = (np.log(p + epsilon) * x) + (np.log(1 - p + epsilon) * (1 - x))
        return np.exp(np.sum(z) / len(z))

    @staticmethod
    def minimize_me(p, a):
        return abs((p * np.log(p)) + ((1 - p) * np.log(1 - p)) - np.log(a))

    @classmethod
    def get_w(cls, a, guess=0.5, bounds=[(0.5, 0.999)]):
        res = minimize(cls.minimize_me, guess, args=(a,), options={'ftol': 0, 'gtol': 1e-09}, method='L-BFGS-B', bounds=bounds)
        return res.x[0]

    @classmethod
    def calculate_imv(cls, y_basic, y_enhanced, y):
        ll_basic = cls.ll(y, y_basic)
        ll_enhanced = cls.ll(y, y_enhanced)
        w0 = cls.get_w(ll_basic)
        w1 = cls.get_w(ll_enhanced)
        return (w1 - w0) / w0

    def calculate_imv_score(self, model_basic, model_enhanced, X_basic, X_enhanced, y):
        if self.model_type == 'classification':
            pred_basic = model_basic.predict_proba(X_basic)[:, 1]
            pred_enhanced = model_enhanced.predict_proba(X_enhanced)[:, 1]
        elif self.model_type == 'regression':
            pred_basic = model_basic.predict(X_basic)
            pred_enhanced = model_enhanced.predict(X_enhanced)
        else:
            raise ValueError("model_type must be 'classification' or 'regression'")
        return self.calculate_imv(pred_basic, pred_enhanced, y)

    def compute_imv_method(self, combination):
        X = self.data[list(combination) + ['constant']]
        y = self.data[self.outcome_variable]
        imv_scores = []
        fold_imv_scores = []

        if self.split_method == 'train_test_split':
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.prop_test, random_state=self.random_seed)
            model_basic = self.model_creator()
            model_enhanced = self.model_creator()
            model_basic.fit(X_train[['constant']], y_train)
            model_enhanced.fit(X_train, y_train)
            imv_score = self.calculate_imv_score(model_basic, model_enhanced, X_test[['constant']], X_test, y_test)
            imv_scores.append(imv_score)
            fold_imv_scores.append(imv_score)
        elif self.split_method == 'kfold':
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_seed)
            for train_index, test_index in kf.split(X):
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
        combinations_list = [
            subset
            for L in range(0, len(self.optional_explanatory_variables) + 1)
            for subset in combinations(self.optional_explanatory_variables, L)
        ]

        with tqdm_joblib(tqdm(total=len(combinations_list))) as progress_bar:
            results = Parallel(n_jobs=2)(
                delayed(self.compute_imv_method)(subset)
                for subset in combinations_list
            )

        self.all_combinations_imv = {combination: (imv_score, fold_scores) for combination, imv_score, fold_scores in results}
        best_combination, (max_imv, best_fold_scores) = max(self.all_combinations_imv.items(), key=lambda item: item[1][0])
        print(f"Best explanatory variables' combination: {best_combination}, with the highest IMV score: {max_imv}")


    @staticmethod
    def calculate_weight(s_size, n):
        return factorial(s_size) * factorial(n - s_size - 1) / factorial(n)

    def calculate_imvshapley_value(self, variable):
        n = max(len(combination) for combination in self.all_combinations_imv.keys())
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
        print(f"SHAP-IMV value for variable {variable}: {imvshapley_value}")
        return imvshapley_value


    def evaluate_imvshapley(self, ax=None, figsize=(12, 4)):
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

        sns.barplot(x=sorted_values, y=sorted_keys, palette=colors, orient='h', ax=ax)

        for index, value in enumerate(sorted_values):
            ax.text(value, index, str(value), va='center')

        ax.set_xlabel('IMVShapley Value')
        ax.set_title('IMVShapley Values')

        if created_ax:
          return fig, ax

        return ax
