import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, log_loss, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from typing import Tuple, Dict

class ModelEvaluator:
    """
        Initializes the ModelEvaluator object with specified target variable and models.

        Parameters:
        target_variable (str): The target variable for model evaluation.
        models (dict): A dictionary of models to be evaluated. Defaults to None, in which case
                       a predefined set of models is used.

        Returns:
        None
        """
        
    def __init__(self, target_variable = 'flag_new_orsyshelf', models = None
    ):
        
        if models == None:
            models = {
            "Logistic Regression": LogisticRegression(random_state=0, max_iter=1_000_000),
            "Random Forest": RandomForestClassifier(random_state=0),
            "Gradient Boosting": GradientBoostingClassifier(random_state=0),
            "Support Vector Machine": SVC(probability=True, random_state=0),
        }
        
        self.target_variable = target_variable
        self.models = models
        self.cv_scores = {}
        self.test_scores = {}
        self.fitted = False
        self.plot_data = {}


    def __prepare_data(self, data) -> Tuple[pd.DataFrame, pd.Series]:

        """
        Prepares the data for model training and evaluation by separating features and target variable.

        Parameters:
        data (pd.DataFrame): The dataset to be used for model evaluation.

        Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple where the first element is a DataFrame of features and 
                                        the second element is a Series of the target variable.
        """

        try:
            exclude_variables = ['cust_id', 'district', 'branch_office', 'bo_highest_sales']
            data_num = data.drop(columns=exclude_variables, axis=1)
        except KeyError as e:
            raise e

        try:
            X = data_num.drop([self.target_variable], axis=1)
            y = data_num[self.target_variable]
        except KeyError:
            raise ValueError(f"A valid target variable is required! {self.target_variable} not present in the dataframe")

        return X, y


    def fit(self, data, data_name=None, test_size=0.3, random_state=42, n_splits=10, n_jobs=-2) -> None:
        """
        Fits the models to the training data and evaluates them using cross-validation and test data.

        Parameters:
        data (pd.DataFrame): The dataset to train and test the models.
        data_name (str): Name of the dataset for plot titles. Defaults to None.
        test_size (float): Proportion of the dataset to include in the test split. Defaults to 0.3.
        random_state (int): Controls the shuffling applied to the data before applying the split. Defaults to 42.
        n_splits (int): Number of folds for cross-validation. Defaults to 10.
        n_jobs (int): The number of CPUs to use to do the computation. Defaults to -2.

        Returns:
        None
        """
        
        if data_name == None:
            self.plot_title = None
        
        self.plot_title = data_name
        
        X, y = self.__prepare_data(data)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=random_state, test_size=test_size, shuffle=True
        )

        kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
        scoring = {
            "AUC": "roc_auc",
            "F1": "f1",
            "LogLoss": "neg_log_loss",
        }

        for name, model  in self.models.items():
            cv_results = cross_validate(
                model, X_train, y_train, cv=kf, scoring=scoring, n_jobs=n_jobs
            )
            self.cv_scores[name] = {
                "AUC": np.mean(cv_results["test_AUC"]),
                "F1": np.mean(cv_results["test_F1"]),
                "Log Loss": -np.mean(cv_results["test_LogLoss"]),
            }

            model.fit(X_train, y_train)
            test_probabilities = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, test_probabilities)
            
            # remove warnings for 0 divisions
            np.seterr(divide='ignore', invalid='ignore')
            f1_scores = 2 * precision * recall / (precision + recall)
            np.seterr(divide='warn', invalid='warn')  
            # accounting for 0 divisions
            f1_scores = np.nan_to_num(f1_scores) 
            
            self.plot_data[name] = {
                "precision": precision,
                "recall": recall,
                "f1_scores": f1_scores
            }

            self.test_scores[name] = {
                "AUC": roc_auc_score(y_test, test_probabilities),
                "F1": f1_score(y_test, model.predict(X_test)),
                "Log Loss": log_loss(y_test, test_probabilities),
            }

        self.fitted = True


    def get_plots(self) -> plt.figure:
        """
        Generates precision-recall and F1-score plots for each model on the test set.

        Returns:
        plt.Figure: A matplotlib figure containing the plots.
        """

        if not self.fitted:
            raise Exception("Fit method must be called before get_plot")

        def plot_function():
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

            for name, data in self.plot_data.items():
                ax1.plot(data['recall'], data['precision'], label=f"{name}")
                ax2.plot(data['recall'], data['f1_scores'], label=f"{name}", linestyle="--")

            ax1.set_title("Precision-Recall Curve")
            ax1.set_xlabel("Recall")
            ax1.set_ylabel("Precision")
            ax1.legend()

            ax2.set_title("F1-Score by Recall")
            ax2.set_xlabel("Recall")
            ax2.set_ylabel("F1-Score")
            ax2.legend()

            #if self.plot_title is not None:
            #    plt.suptitle(f"Model Evaluation Plot for {self.plot_title}", fontsize=16)
            #else:
            #    plt.suptitle("Model Evaluation Plot", fontsize=16)

            plt.tight_layout()

            return plt

        return plot_function


    def get_scores(self, only_test: bool = True) -> Dict:
        """
        Returns the evaluation scores of the models.

        Parameters:
        only_test (bool): If True, returns only test scores; if False, returns both test and cross-validation scores.
                          Defaults to True.

        Returns:
        dict: A dictionary containing the evaluation scores.
        """
        
        if not self.fitted:
            raise Exception("Fit method must be called before display_scores")

        if only_test:
            return self.test_scores
        else:
            return {"Test Scores": self.test_scores, "CV Scores": self.cv_scores}
        
        
    

class ModelFitter:
    def __init__(self, target_variable='flag_new_orsyshelf'):
        """
        Initializes the ModelFitter object with the specified target variable.

        Parameters:
        target_variable (str): The target variable for model fitting. Defaults to 'flag_new_orsyshelf'.

        Returns:
        None
        """

        self.target_variable = target_variable


    def __prepare_data(self, data, random_state) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepares the data for model fitting by separating features and the target variable, and performing a train-test split.

        Parameters:
        data (pd.DataFrame): The dataset to be used for model fitting.
        random_state (int): Controls the shuffling applied to the data before applying the split.

        Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple where the first element is a DataFrame of training features and 
                                        the second element is a Series of the training target variable.
        """

        exclude_variables = ["district", "branch_office", "bo_highest_sales", "cust_id"]
        
        features = data.drop(columns=exclude_variables + [self.target_variable], axis=1)
        target = data[self.target_variable]
        X_train, _, y_train, _ = train_test_split(
            features, target, test_size=0.3, random_state=random_state
        )
        return X_train, y_train


    def grid_fit(self, data, params, cv=5, random_state=1860, verbose=4, n_jobs= -2) -> Tuple[Dict, float]:
        """
        Fits a Gradient Boosting Classifier to the training data using Grid Search for hyperparameter tuning.

        Parameters:
        data (pd.DataFrame): The dataset to be used for fitting the model.
        params (dict): Hyperparameters to be used in Grid Search.
        cv (int): Number of folds for cross-validation. Defaults to 5.
        random_state (int): Controls the randomness of the Gradient Boosting Classifier. Defaults to 1860.
        verbose (int): Controls the verbosity of Grid Search. Defaults to 4.
        n_jobs (int): The number of CPUs to use to do the computation. Defaults to -2.

        Returns:
        Tuple[Dict, float]: A tuple containing the best parameters from Grid Search and the best score achieved.
        """
        
        X_train, y_train = self.__prepare_data(data, random_state)
        
        grid = GridSearchCV(
            GradientBoostingClassifier(random_state=random_state),
            params,
            cv=cv,
            scoring="f1",
            verbose=verbose,
            n_jobs=n_jobs,
        )

        grid.fit(X_train, y_train)

        best_params = grid.best_params_
        best_score = grid.best_score_

        return best_params, best_score
