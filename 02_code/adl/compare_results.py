import pandas as pd
from typing import List

def compare_results(dataframes: List[pd.DataFrame], df_names: List[str], identifier='cust_id') -> pd.DataFrame:
    """
    Compares multiple pandas DataFrames to find common values in a specified identifier column. 
    It then generates a new DataFrame indicating the presence of these common values in each original DataFrame, along with their counts.

    Parameters:
    dataframes (List[pd.DataFrame]): A list of pandas DataFrames to be compared.
    df_names (List[str]): A list of names corresponding to each DataFrame in 'dataframes'. These names are used as flags in the resulting DataFrame.
    identifier (str): The column name in each DataFrame to be used as the identifier for comparison. Defaults to 'cust_id'.

    Returns:
    pd.DataFrame: A DataFrame containing the common values in the identifier column across all input DataFrames. It includes flags for each 
    DataFrame to indicate the presence of these common values, along with a count of their occurrences across all DataFrames.
    """

    if len(dataframes) != len(df_names):
        raise ValueError("The length of dataframes and names lists must be the same.")

    # Initialize an empty list to store processed dataframes
    processed_dfs = []

    # Iterate over each dataframe and its name
    for df, name in zip(dataframes, df_names):
        # Reset the index if it's a MultiIndex or set a simple range index
        df = df.reset_index(drop=True)

        # Add a source column for each dataframe
        df['source'] = name
        processed_dfs.append(df)

    # Combine all processed dataframes
    combined = pd.concat(processed_dfs)

    # Get dummies for the source column
    source_dummies = pd.get_dummies(combined['source'])

    # Concatenate the dummies with the combined dataframe
    combined = pd.concat([combined, source_dummies], axis=1)

    # Group by the identifier and sum up the source dummy columns
    common_values = combined.groupby(identifier)[source_dummies.columns].sum()

    # Add a count column for the total occurrences
    common_values['Count'] = common_values.sum(axis=1)

    # Reset index to turn the series into a dataframe
    common_values.reset_index(inplace=True)

    # Reorder columns to have 'Count' as the second column
    columns = [identifier, 'Count'] + df_names
    common_values = common_values[columns]

    # Sort the dataframe by 'Count' in descending order and reset the index
    common_values = common_values.sort_values(by='Count', ascending=False).reset_index(drop=True)

    return common_values

