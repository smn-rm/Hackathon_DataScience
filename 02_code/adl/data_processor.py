import pandas as pd
import numpy as np
from typing import List, Callable, Dict
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataProcessor:
    # TODO change mode like scale

    def __init__(self, file_path: str, remove_multicollinearity = True):
        """
        Initializes the DataProcessor object by loading data from a specified file path 
        and optionally removes multicollinearity.

        Parameters:
        file_path (str): The file path of the dataset to be processed.
        remove_multicollinearity (bool): If True, variables leading to multicollinearity 
        are removed from the dataset.
        """
        
        self.data = self.__clean_data(file_path, remove_multicollinearity)



    def __clean_data(self, file_path: str, remove_multicollinearity) -> pd.DataFrame:
        """
        Cleans the dataset by loading it from the specified file path and applying various data 
        preprocessing steps like filtering, adding new columns, and handling multicollinearity.

        Parameters:
        file_path (str): The file path of the dataset to be processed.
        remove_multicollinearity (bool): If True, multicollinearity is removed from the dataset.

        Returns:
        pd.DataFrame: The cleaned and optionally multicollinearity-removed dataset.
        """
        
        data = pd.read_csv(file_path, index_col=False)        

        def __insert_negative_indicator_columns(data, prefix):
            """Insert columns indicating negative values for specified columns."""
            for col in [c for c in data.columns if c.startswith(prefix)]:
                neg_col_name = col + "_neg"
                data.insert(data.columns.get_loc(col) + 1, neg_col_name, data[col].apply(lambda x: 1 if x < 0 else 0))


        # Filter 'dunning_level_current' and 'dunning_level_highest' between 0 and 5
        dunning_filters = ((data["dunning_level_current"] >= 0) & (data["dunning_level_current"] <= 5) &
                        (data["dunning_level_highest"] >= 0) & (data["dunning_level_highest"] <= 5))
        data = data[dunning_filters]
        
        # Add 'dunning_level_max' column
        data.insert(21, "dunning_level_max", data[["dunning_level_current", "dunning_level_highest"]].max(axis=1))
        # removing highest, as we build max
        data = data.drop(columns=['dunning_level_highest'], axis=1)
        
        # Insert negative indicator columns for 'rev_' columns
        __insert_negative_indicator_columns(data, "rev_")

        data.insert(20, "sales_relevant_share", (data["sales_orsy_relevant"] / data["sales"]) * 100)

        
        # Add 'emp_count_lvl' column
        data = data[data["emp_count"] > 0]
        data.insert(6, "emp_count_lvl", data["emp_count"].apply(lambda x: 0 if x <= 3 else (1 if x <= 9 else 2)))


        upload_date = pd.to_datetime("2023-10-24")  
        data["last_buy"] = (upload_date - pd.to_datetime(data["last_buy"])).dt.total_seconds() / (60 * 60 * 24)
        data["cust_since"] = (upload_date - pd.to_datetime(data["cust_since"])).dt.total_seconds() / (60 * 60 * 24)
        data.rename(columns={'last_buy': 'td_last_buy', 'cust_since': 'td_cust_since'}, inplace=True)

        data["share_buyweeks"] = data["count_buyweeks"] / (data["td_cust_since"] / 7)
        data["share_buydays"] = data["count_buydays"] / (data["td_cust_since"])

            
        
        
        # convert nominal /ordinal variables to dummies
        # List of variables to exclude when conducting the analysis with the models
        include_variables = ['cust_id', 'district', 'branch_office', 'bo_highest_sales']
        subset_data = data.drop(columns=include_variables, axis=1)

        subset_data = pd.get_dummies(subset_data, drop_first=True) # converts strings
        subset_data = pd.get_dummies(subset_data, columns=['region', 'market_seg', 'ccp_most_used'], drop_first=True) # converts nominal/ordinals
        
        dataframe = pd.concat([data[include_variables], subset_data], axis=1)
        
        if remove_multicollinearity:
            dataframe = dataframe.drop(columns=['sales', 'count_buydays', 'count_buyweeks', 'count_buymonths'], axis=1)

        return dataframe
    
    
    
    def __extract_correlated_features(self, dataframe, target_variable, threshold) -> pd.DataFrame:
        """
        Extracts features from a given dataframe that are correlated above a certain threshold with a target variable.

        Parameters:
        dataframe (pd.DataFrame): The dataframe from which to extract features.
        target_variable (str): The target variable against which to measure correlation.
        threshold (float): The correlation threshold above which features are selected.

        Returns:
        pd.DataFrame: A DataFrame containing features with high correlation to the target variable.
        """

        include_variables = ['cust_id', 'district', 'branch_office', 'bo_highest_sales']
  
        subset_data = dataframe.drop(columns=include_variables, axis=1)

        correlation_matrix = subset_data.corr()
        correlations = correlation_matrix[target_variable]
        corr_features = correlations[np.abs(correlations) >= threshold].index.tolist()
        
        
        df_corr_features = pd.concat([dataframe[include_variables], subset_data[corr_features]], axis=1)
        
        return df_corr_features



    def __remove_rows_by_conditions(self, dataframe, conditions) -> pd.DataFrame:
        """
        Removes rows from the DataFrame based on a specified condition. Rows for which the condition 
        function returns True will be excluded from the resulting DataFrame.

        Parameters:
        dataframe (pd.DataFrame): The dataframe from which rows are to be removed.
        conditions (Callable[[pd.DataFrame], pd.Series]): A function that takes the DataFrame as input
          and returns a pd.Series of booleans, where True indicates rows to be excluded.

        Returns:
        pd.DataFrame: A new DataFrame with rows filtered out based on the condition.
        """

        df = dataframe.copy()
        
        try:
            df = df[~conditions(df)]
        except KeyError as e:
            raise KeyError(f"Key error encountered in the condition function. Details: {e}")

        return df



    def __standardize_data(self, dataframe: pd.DataFrame, unbiased: bool) -> pd.DataFrame:
        """
        Standardizes the non-dummy numeric columns of the dataframe using either unbiased standard deviation 
        or a standard scaler.

        Parameters:
        dataframe (pd.DataFrame): The dataframe to be standardized.
        unbiased (bool): If True, uses unbiased standard deviation for scaling.

        Returns:
        pd.DataFrame: The standardized DataFrame.
        """

        scaler = StandardScaler()
        
        include_variables = ['cust_id', 'district']
        subset_data = dataframe.drop(columns=include_variables, axis=1)
        numeric_columns = subset_data.select_dtypes(include=['number']).columns
        non_dummy_columns = [col for col in numeric_columns if subset_data[col].nunique() > 2]

        df = pd.concat([dataframe[include_variables], subset_data], axis=1)

        # include variabES
        if unbiased:
            means = df[non_dummy_columns].mean()
            stds = df[non_dummy_columns].std(ddof=1)
            df[non_dummy_columns] = (df[non_dummy_columns] - means) / stds
            print("Unbiased standardizing applied")

        else:
            df[non_dummy_columns] = scaler.fit_transform(df[non_dummy_columns])
            print("Biased standardizing applied")
                        

        return df



    def __normalize_data(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes the non-dummy numeric columns of the dataframe using Min-Max scaling.

        Parameters:
        dataframe (pd.DataFrame): The dataframe to be normalized.

        Returns:
        pd.DataFrame: The normalized DataFrame.
        """
        
        scaler = MinMaxScaler()
        
        include_variables = ['cust_id', 'district']
        subset_data = dataframe.drop(columns=include_variables, axis=1)
        numeric_columns = subset_data.select_dtypes(include=['number']).columns
        non_dummy_columns = [col for col in numeric_columns if subset_data[col].nunique() > 2]

        df = pd.concat([dataframe[include_variables], subset_data], axis=1)

        df[non_dummy_columns] = scaler.fit_transform(df[non_dummy_columns])
            
        return df
    
    
    
    def process_data(self, mode: str = [None, 'extract', 'remove', 'both'], scale: str = ['standardize', 'normalize', None], unbiased: bool = True,
                     target_variable: str = 'flag_new_orsyshelf', threshold: float = 0.2, 
                     conditions: Callable[[pd.DataFrame], pd.Series] = None) -> pd.DataFrame:
        """
        Processes the class's data based on the specified mode, including feature extraction, row removal, 
        and data scaling.

        Parameters:
        mode (str): The processing mode. Can be 'None', 'extract', 'remove', or 'both'.
        scale (str): The scaling method to use. Can be 'standardize', 'normalize', or 'None'.
        unbiased (bool): If True and standardizing, uses unbiased standard deviation for scaling.
        target_variable (str): Target variable for feature extraction.
        threshold (float): Correlation threshold for feature extraction.
        conditions (Callable[[pd.DataFrame], pd.Series]): Conditions for row removal.

        Returns:
        pd.DataFrame: The processed DataFrame.
        """

        
        if mode == [None, 'extract', 'remove', 'both']:
            warnings.warn(f"No mode specified. Defaults to {None}.")
        
        elif mode not in [None, 'extract', 'remove', 'both']:
            raise ValueError(f"Invalid mode '{mode}'. Choose one of {[None, 'extract', 'remove', 'both']}.")
        
        if scale ==  ['standardize', 'normalize', None]:
            warnings.warn(f"Nothing specified. Data will be standardized")
        
        
        if mode == None or mode == [None, 'extract', 'remove', 'both']:
            processed_data = self.data
            
        elif mode == 'extract':
            if target_variable is None or threshold is None:
                raise ValueError("target_variable and threshold must be provided for extraction.")
            processed_data = self.__extract_correlated_features(self.data, target_variable, threshold)
        
        elif mode == 'remove':
            if conditions is None:
                raise ValueError("conditions_dict must be provided for removal.")
            processed_data = self.__remove_rows_by_conditions(self.data, conditions)
            
        elif mode == 'both':
            if target_variable is None or threshold is None or conditions is None:
                raise ValueError("All parameters must be provided for 'both' mode.")
            correlated_data = self.__remove_rows_by_conditions(self.data, conditions)
            processed_data = self.__extract_correlated_features(correlated_data, target_variable, threshold)

        if scale == ['standardize', 'normalize', None] or scale == 'standardize':
            processed_data = self.__standardize_data(processed_data, unbiased)
        elif scale == 'normalize':
            print("Normalizing applied")
            processed_data = self.__normalize_data(processed_data)
        else:
            print('No scaling applied')
            
          
        return processed_data
