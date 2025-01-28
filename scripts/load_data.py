# scripts/load_data.py
# Import necessary libraries
import pandas as pd

# Load data function
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file and return a DataFrame.
    
    Parameters:
    -----------
    file_path : str
        The path to the dataset file.
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataset in a pandas DataFrame format.
    """
    try:
        df = pd.read_csv(file_path, index_col=0)
        print(f"Data successfully loaded from {file_path}")
        print(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.\n")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return pd.DataFrame()  # Return empty DataFrame in case of failure