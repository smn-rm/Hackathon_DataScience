import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings 
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from typing import List, Tuple

class AnomalyDetector:
    def __init__(self, target_variable: str = 'flag_new_orsyshelf', identifier: str = 'cust_id'):
        """
        Initializes the AnomalyDetector object with a specified target variable and identifier.

        Parameters:
        target_variable (str): The name of the target variable in the dataset, default 'flag_new_orsyshelf'.
        identifier (str): The name of the identifier column in the dataset, default 'cust_id'.
        """
        self.target_variable = target_variable
        self.identifier = identifier


    def __prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares the data for anomaly detection by selecting numerical columns and excluding object type columns.

        Parameters:
        data (pd.DataFrame): The DataFrame containing the data to be prepared.

        Returns:
        pd.DataFrame: The prepared DataFrame with only numerical columns.
        """
        return data.select_dtypes(exclude=['object'])


    def __fit_and_detect(self, data, contamination_levels, random_states) -> pd.DataFrame:
        """
        Fits an Isolation Forest model and detects anomalies in the dataset, iterating over various contamination levels 
        and random states.

        Parameters:
        data (pd.DataFrame): The DataFrame containing the data for anomaly detection.
        contamination_levels (List[float]): A list of contamination levels to use in the Isolation Forest.
        random_states (List[int]): A list of random states for reproducibility.

        Returns:
        pd.DataFrame: A DataFrame with detected anomalies along with their scores and predictions.
        """
        
        all_results = []
        prepared_data = self.__prepare_data(data)
        
        for contamination_level in contamination_levels:
            for random_state in random_states:
                model = IsolationForest(contamination=contamination_level, random_state=random_state)
                
                model.fit(prepared_data)

                # Detection
                anomaly_score = model.decision_function(prepared_data)
                predictions = model.predict(prepared_data)

                # Create a copy of the data to avoid modifying the original DataFrame
                temp_data = data.copy()
                temp_data['anomaly_score'] = anomaly_score
                temp_data['predictions'] = predictions

                anomalies = temp_data[(temp_data[self.target_variable] == 0) & (temp_data['anomaly_score'] < 0)]
                all_results.append(anomalies[[self.identifier, 'anomaly_score']])

        all_results_df = pd.concat(all_results, ignore_index=True)
        return all_results_df


    def extract_anomalies(
        self, data: pd.DataFrame, contamination_levels: List[float] = [0.01, 0.02, 0.03, 0.04, 0.05], 
        random_states: List[int] = range(100), count_threshold: int = 250
    ) -> pd.DataFrame:
        """
        Extracts anomalies from the data using Isolation Forest, applying various contamination levels and random states.
        It returns customers with anomaly counts exceeding a specified threshold.

        Parameters:
        data (pd.DataFrame): The DataFrame containing the data for anomaly detection.
        contamination_levels (List[float]): A list of contamination levels to use in the Isolation Forest.
        random_states (List[int]): A list of random states for reproducibility.
        count_threshold (int): The minimum count of anomalies required to consider a customer as an outlier.

        Returns:
        pd.DataFrame: A DataFrame of customers exceeding the anomaly count threshold.
        """
        
        anomalies_df = self.__fit_and_detect(data, contamination_levels, random_states)
        cust_id_counts = anomalies_df[self.identifier].value_counts()
        cust_id_counts_df = cust_id_counts.reset_index()
        cust_id_counts_df.columns = [self.identifier, 'count']
        return cust_id_counts_df[cust_id_counts_df['count'] >= count_threshold]
    
    # ! not used
    def calculate_total_count_for_selected_cust_ids(self, dataframes: List[pd.DataFrame], count_threshold: int = 250) -> pd.DataFrame:
        """
        Calculates the total count of anomalies for selected customer IDs across multiple dataframes,
        filtering those exceeding a specified threshold.

        Parameters:
        dataframes (List[pd.DataFrame]): A list of DataFrames containing customer ID counts.
        count_threshold (int): The threshold for the minimum count of anomalies.

        Returns:
        pd.DataFrame: A DataFrame with total anomaly counts for each customer ID.
        """

        filtered_dfs = []
        for df in dataframes:
            filtered_df = df[df['count'] > count_threshold]
            filtered_dfs.append(filtered_df)

        if not filtered_dfs:
            return pd.DataFrame()

        merged_df = filtered_dfs[0]
        for i, filtered_df in enumerate(filtered_dfs[1:], start=2):
            suffix = f'_{i}'
            merged_df = pd.merge(merged_df, filtered_df[[self.identifier, 'count']], on=self.identifier, suffixes=('', suffix))

        #final_cust_ids = merged_df[self.identifier].unique()
        total_count = merged_df.groupby(self.identifier)[['count'] + [f'count_{i}' for i in range(2, len(filtered_dfs) + 1)]].sum().reset_index()
        return total_count




class CosineSimilarityCalculator:
    def __init__(self, group_variable: str = 'flag_new_orsyshelf', identifier: str = 'cust_id'):
        """
        Initializes the CosineSimilarityCalculator with a specified group variable and identifier.

        Parameters:
        group_variable (str): The name of the column used to group the data. Default is 'flag_new_orsyshelf'.
        identifier (str): The name of the column used as an identifier in the similarity analysis. Default is 'cust_id'.
        """
        
        self.group_variable = group_variable
        self.identifier = identifier


    def __prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """
        Prepares the data by splitting it into two groups based on the group variable and selects the relevant columns for analysis.

        Parameters:
        data (pd.DataFrame): The DataFrame containing the data to be prepared.

        Returns:
        Tuple: A tuple containing DataFrames for customers, non-customers, and non-customer identifiers.
        """

        customers = data[data[self.group_variable] == 1].drop(self.group_variable, axis=1).iloc[:, 4:]
        non_customers = data[data[self.group_variable] == 0].drop(self.group_variable, axis=1).iloc[:, 4:]
        non_customer_ids = data[data[self.group_variable] == 0][self.identifier].values
        return customers, non_customers, non_customer_ids


    def __calculate_cosine_similarity(self, non_customers, customers) -> np.ndarray:
        """
        Calculates the cosine similarity matrix between two sets of data.

        Parameters:
        non_customers (pd.DataFrame): The DataFrame representing the non-customers.
        customers (pd.DataFrame): The DataFrame representing the customers.

        Returns:
        ndarray: The calculated cosine similarity matrix.
        """

        return cosine_similarity(non_customers.values, customers.values)


    def get_cosine_average(self, data, n_best: int = 500) -> pd.DataFrame:
        """
        Calculates and returns the average cosine similarity for each non-customer.

        Parameters:
        data (pd.DataFrame): The DataFrame containing the data for similarity calculation.
        n_best (int): The number of top results to return, based on the highest average cosine similarity.

        Returns:
        pd.DataFrame: A DataFrame containing the identifiers and average cosine similarity for each non-customer.
        """

        if n_best < 1 or n_best > data.shape[0]:
            raise ValueError("'n_best' must be a positive int and smaller than the rows of the dataframe!")
        
        customers, non_customers, non_customer_ids = self.__prepare_data(data)
        cosine_sim_matrix = self.__calculate_cosine_similarity(non_customers, customers)

        cosine_avg = pd.DataFrame(cosine_sim_matrix.mean(axis=1), columns=['average'])
        cosine_avg[self.identifier] = non_customer_ids
        sorted_avg = cosine_avg[[self.identifier, 'average']].sort_values(by='average', ascending=False)
        return sorted_avg.head(n_best) if n_best else sorted_avg


    def get_cosine_count(self, data: pd.DataFrame, threshold=0.5, n_best: int = 500) -> pd.DataFrame:
        """
        Calculates and returns the count of cosine similarity values above a specified threshold for each non-customer.

        Parameters:
        data (pd.DataFrame): The DataFrame containing the data for similarity calculation.
        threshold (float): The threshold for counting cosine similarities.
        n_best (int): The number of top results to return, based on the highest count of similarities above the threshold.

        Returns:
        pd.DataFrame: A DataFrame containing the identifiers and count of high cosine similarities for each non-customer.
        """
        
        if n_best < 1 or n_best > data.shape[0]:
            raise ValueError("'n_best' must be a positive int and smaller than the rows of the dataframe!")
        
        customers, non_customers, non_customer_ids = self.__prepare_data(data)
        cosine_sim_matrix = self.__calculate_cosine_similarity(non_customers, customers)

        cosine_count = pd.DataFrame((cosine_sim_matrix > threshold).sum(axis=1), columns=['count'])
        cosine_count[self.identifier] = non_customer_ids
        sorted_count = cosine_count[[self.identifier, 'count']].sort_values(by='count', ascending=False)
        return sorted_count.head(n_best) if n_best else sorted_count
    

    def get_both_cosine_metrics(self, data: pd.DataFrame, 
        threshold=0.5, n_best: int = 500, 
        sort_by: str = ['average', 'count']
    ) -> pd.DataFrame:
        """
        Returns a DataFrame containing both the average and count of cosine similarities for each non-customer.

        Parameters:
        data (pd.DataFrame): The DataFrame containing the data for similarity calculation.
        threshold (float): The threshold for counting cosine similarities.
        n_best (int): The number of top results to return.
        sort_by (str): The metric by which to sort the results, either 'average' or 'count'.

        Returns:
        pd.DataFrame: A DataFrame containing the identifiers, average, and count of cosine similarities for each non-customer.
        """

        
        if n_best < 1 or n_best > data.shape[0]:
            raise ValueError("'n_best' must be a positive int and smaller than the rows of the dataframe!")
        
        if sort_by == ['average', 'count']:
            sort_by = 'average'
            warnings.warn("Nothing chosen to sort by. Will be sorted by 'average'")
        elif sort_by != 'average' and sort_by != 'count':
            raise ValueError("sort_by must be either 'average' or 'count'")
        
        avg_df = self.get_cosine_average(data)
        count_df = self.get_cosine_count(data, threshold)
        merged_df = pd.merge(avg_df, count_df, on=self.identifier).sort_values(by=sort_by, ascending=False).reset_index()
        return  merged_df[[self.identifier, 'average', 'count']].head(n_best)




class PropensityScorer:
    def __init__(self, target_variable = 'flag_new_orsyshelf', identifier = 'cust_id'):
        """
        Initializes the PropensityScorer with a target variable and an identifier.

        Parameters:
        target_variable (str): The name of the target variable in the dataset.
        identifier (str): The name of the identifier column in the dataset.
        """
        
        self.target_variable = target_variable
        self.identifier = identifier


    def __prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepares the data by selecting numerical columns and separating features from the target variable.

        Parameters:
        data (pd.DataFrame): The DataFrame containing the data to be prepared.

        Returns:
        Tuple: A tuple containing feature matrix (X), target vector (y), and the numerical data subset (data_num).
        """

        try:
            exclude_variables = [self.identifier, 'district', 'branch_office', 'bo_highest_sales']
            data_num = data.drop(columns=exclude_variables, axis=1)
        except:
            pass

        try:
            X = data_num.drop([self.target_variable ], axis=1)
            y = data_num[self.target_variable ]
        except ValueError as e:
            raise ValueError(f"A valid target variable is required! {e}")
        
        return X, y, data_num


    def logistic_regression(self, data: pd.DataFrame, C: float = 1.0, 
        penalty: str = 'l2', n_best: int = 10
    ) -> pd.DataFrame:
        """
        Applies logistic regression to predict the propensity scores and returns the top N results.

        Parameters:
        data (pd.DataFrame): The DataFrame containing the data for model fitting.
        C (float): Inverse of regularization strength; must be a positive float.
        penalty (str): Used to specify the norm used in the penalization ('l1', 'l2', 'elasticnet', None).
        n_best (int): Number of top results to return based on the propensity scores.

        Returns:
        pd.DataFrame: A DataFrame containing the top N results with propensity scores.
        """
        
        X, y, data_num = self.__prepare_data(data)
        
        penalties = ['l1', 'l2', 'elasticnet', None]
        if penalty not in penalties:
            raise ValueError(f"'{penalty}' is an invalid penalty! Pick one of these {penalties}.")

        if C < 0:
            raise ValueError("'C' must be a positive float!")

        if n_best < 1 or n_best > data.shape[0]:
            raise ValueError("'n_best' must be a positive int and smaller than the rows of the dataframe!")

        try:
            logistic_reg = LogisticRegression(
                penalty=penalty, C=C, random_state=42, 
                max_iter=1_000_000_000, class_weight='balanced'
            )
            probs_orsy = self.__fit_predict_proba(logistic_reg, X, y, data_num)
        except:
            raise

        return self.__get_best_n(probs_orsy, n_best, data)

    
    def random_forest(self, data: pd.DataFrame, n_estimators: int = 100, max_depth: int = None, 
        min_samples_split: int = 2, min_samples_leaf: int = 1, criterion: str = 'gini', 
        n_best: int = 10
    ) -> pd.DataFrame:
        """
        Applies a random forest classifier to predict the propensity scores and returns the top N results.

        Parameters:
        data (pd.DataFrame): The DataFrame containing the data for model fitting.
        n_estimators (int): The number of trees in the forest.
        max_depth (int): The maximum depth of the tree.
        min_samples_split (int): The minimum number of samples required to split an internal node.
        min_samples_leaf (int): The minimum number of samples required to be at a leaf node.
        criterion (str): The function to measure the quality of a split ('gini', 'entropy').
        n_best (int): Number of top results to return based on the propensity scores.

        Returns:
        pd.DataFrame: A DataFrame containing the top N results with propensity scores.
        """

        X, y, data_num = self.__prepare_data(data)
        
        criteria = ['gini', 'entropy', 'log_loss']
        if criterion not in criteria:
            raise ValueError(f"'{criterion}' is an invalid criterion! Pick one of these {criteria}.")

        if n_estimators < 1:
            raise ValueError("'n_estimators' must be a positive int > 0!")

        if min_samples_split < 1:
            raise ValueError("'min_samples_split' must be a positive int > 0!")

        if min_samples_leaf < 1:
            raise ValueError("'min_samples_leaf' must be a positive int > 0!")

        if n_best < 0 or n_best > data.shape[0]:
            raise ValueError("'n_best' must be a positive int and smaller than the rows of your dataframe!")

        try:
            random_forest = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                   min_samples_split=min_samples_split,
                                                   min_samples_leaf=min_samples_leaf, criterion=criterion,
                                                   random_state=42)
            probs_orsy = self.__fit_predict_proba(random_forest, X, y, data_num)
        except:
            raise

        return self.__get_best_n(probs_orsy, n_best, data)


    def gradient_boosting(self, data: pd.DataFrame, loss: str = 'log_loss', 
        learning_rate: float = 0.1, n_estimators: int = 100, criterion: str = 'friedman_mse', 
        min_samples_split: int = 2, min_samples_leaf: int = 1,
        max_depth: int = 3, n_best: int = 10
    ) -> pd.DataFrame:
        """
        Applies a gradient boosting classifier to predict the propensity scores and returns the top N results.

        Parameters:
        data (pd.DataFrame): The DataFrame containing the data for model fitting.
        loss (str): Loss function to be optimized ('log_loss', 'exponential').
        learning_rate (float): Learning rate shrinks the contribution of each tree.
        n_estimators (int): The number of boosting stages to be run.
        criterion (str): The function to measure the quality of a split ('friedman_mse', 'squared_error').
        min_samples_split (int): The minimum number of samples required to split an internal node.
        min_samples_leaf (int): The minimum number of samples required to be at a leaf node.
        max_depth (int): Maximum depth limits the number of nodes in the tree.
        n_best (int): Number of top results to return based on the propensity scores.

        Returns:
        pd.DataFrame: A DataFrame containing the top N results with propensity scores.
        """
        
        X, y, data_num = self.__prepare_data(data)
        
        losses = ['log_loss', 'exponential']
        if loss not in losses:
            raise ValueError(f"'{loss}' is an invalid criterion! Pick one of these {losses}.")

        if learning_rate < 0:
            raise ValueError("'learning_rate' must be a positive float > 0!")

        if n_estimators < 1:
            raise ValueError("'n_estimators' must be a positive int > 0!")

        criteria = ['friedman_mse', 'squared_error']
        if criterion not in criteria:
            raise ValueError(f"'{criterion}' is an invalid criterion! Pick one of these {criteria}.")

        if min_samples_split < 0:
            raise ValueError("'min_samples_split' must be positive!")

        if min_samples_leaf < 0:
            raise ValueError("'min_samples_leaf' must be positive!")

        if n_best < 0 or n_best > data.shape[0]:
            raise ValueError("'n_best' must be a positive int and smaller than the rows of your dataframe!")

        try:
            gradient_boosting = GradientBoostingClassifier(n_estimators=n_estimators, criterion=criterion,
                                                           learning_rate=learning_rate, max_depth=max_depth,
                                                           min_samples_split=min_samples_split,
                                                           min_samples_leaf=min_samples_leaf, random_state=42)
            probs_orsy = self.__fit_predict_proba(gradient_boosting, X, y, data_num)
        except:
            raise

        return self.__get_best_n(probs_orsy, n_best, data)


    def __fit_predict_proba(self, model, X, y, data_num) -> np.ndarray:
        """
        Fits the model to the data and predicts the probability of the target class.

        Parameters:
        model: The machine learning model to be fitted.
        X: Feature matrix for model fitting.
        y: Target vector for model fitting.
        data_num: The numerical data subset for prediction.

        Returns:
        ndarray: An array containing the predicted probabilities.
        """
        
        try:
            model.fit(X, y)

            df_non_orsy = data_num[data_num[self.target_variable] == 0]
            x_non_orsy = df_non_orsy.drop([self.target_variable], axis=1)
            probs_orsy = model.predict_proba(x_non_orsy)[:, 1]
        except:
            raise

        return probs_orsy


    def __get_best_n(self, probs, n, data) -> pd.DataFrame:
        """
        Returns the top N data points with the highest predicted probabilities.

        Parameters:
        probs (ndarray): An array containing the predicted probabilities.
        n (int): Number of top results to return.
        data (pd.DataFrame): The original DataFrame to extract the top results from.

        Returns:
        pd.DataFrame: A DataFrame containing the top N results with the highest probabilities.
        """

        indices_best_n = pd.DataFrame(probs, columns=['probs']).nlargest(n=n, columns="probs", keep='all').index
        df_best_n = data[data['flag_new_orsyshelf'] == 0].iloc[indices_best_n]
        df_best_n['orsy_prob'] = probs[indices_best_n]
        return df_best_n[[self.identifier, 'orsy_prob']]