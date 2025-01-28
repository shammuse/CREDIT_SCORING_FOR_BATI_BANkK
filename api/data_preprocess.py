import pandas as pd
import sys, os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, KBinsDiscretizer

def preprocess_transactions(df):
    """Preprocesses transaction data for model input."""

    # Aggregate features
    aggregate_features = df.groupby('CustomerId').agg(
        Total_Transaction_Amount=('Amount', 'sum'),
        Average_Transaction_Amount=('Amount', 'mean'),
        Transaction_Count=('TransactionId', 'count')
    ).reset_index()

    # Convert 'TransactionStartTime' to datetime, handle missing values
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
    df['TransactionStartTime'] = df['TransactionStartTime'].dt.tz_localize(None) # Make datetime timezone-naive

    # Extract features
    df['Transaction_Hour'] = df['TransactionStartTime'].dt.hour
    df['Transaction_Day'] = df['TransactionStartTime'].dt.day
    df['Transaction_Month'] = df['TransactionStartTime'].dt.month
    df['Transaction_Year'] = df['TransactionStartTime'].dt.year

    # Merge aggregate features
    final_df = df.merge(aggregate_features, on='CustomerId', how='left')

    # Label encoding
    label_columns = ['PricingStrategy']
    label_encoder = LabelEncoder()
    for col in label_columns:
         final_df[col] = label_encoder.fit_transform(final_df[col].astype(str))

    # Numerical columns for scaling
    numerical_columns = final_df.select_dtypes(include=['float64', 'int64']).columns
    min_max_scaler = MinMaxScaler()
    final_df[numerical_columns] = min_max_scaler.fit_transform(final_df[numerical_columns])

    # RFMS calculations
    final_df['Recency'] = final_df.groupby('CustomerId')['Transaction_Year'].transform('max')
    final_df['Frequency'] = final_df['Transaction_Count']
    final_df['Monetary'] = final_df['Total_Transaction_Amount']

    # Normalize RFMS
    rfms_cols = ['Recency', 'Frequency', 'Monetary']
    for col in rfms_cols:
        final_df[col] = (final_df[col] - final_df[col].min()) / (final_df[col].max() - final_df[col].min())
    
    # Compute RFMS Score
    final_df['RFMS_Score'] = final_df[rfms_cols].mean(axis=1)

    # Fill missing values
    for col in final_df.select_dtypes(include=['float64', 'int64']).columns:
        final_df[col] = final_df[col].fillna(0)
    for col in final_df.select_dtypes(include=['object']).columns:
        final_df[col] = final_df[col].fillna(final_df[col].mode()[0])

    # Bin RFMS Score
    n_bins = 5
    kbin = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    final_df['RFMS_Binned'] = kbin.fit_transform(final_df[['RFMS_Score']])

    # Drop unneeded features
    X = final_df.drop(['TransactionId', 'CustomerId', 'TransactionStartTime'], axis=1)
    return X

#import gdown
import pickle
#import os
#
#def download_file_from_google_drive(file_id, destination):
#    """
#    Downloads a file from Google Drive using gdown.
#
#    Args:
#        file_id (str): The Google Drive file ID.
#        destination (str): The local file path where the downloaded file will be saved.
#    """
#    url = f"https://drive.google.com/uc?id={file_id}"
#    
#    # Use gdown to download the file from Google Drive
#    gdown.download(url, destination, quiet=False)


def load_model():
    """
    Checks if the model file exists locally. If not, downloads it from Google Drive.

    Returns:
        The loaded model object.
    """
    # Get the directory of the current script
    current_dir = os.path.dirname(__file__)
    
    # Construct the correct path to the model file
    model_path = os.path.join(current_dir, 'model/best_model.pkl')
    
    # If the model doesn't exist locally, download it
    #if not os.path.exists(model_path):
    #    file_id = '1A2B3C4D5EF6GHIJKL'  # Replace with your actual file ID from Google Drive
    #    download_file_from_google_drive(file_id, model_path)

    # Load the model using pickle
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model
