import pandas as pd
import sys, os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, KBinsDiscretizer
import pickle

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
    df['TransactionStartTime'] = df['TransactionStartTime'].dt.tz_localize(None)  # Make datetime timezone-naive

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

def load_model():
    """
    Loads the trained model from a local file.

    Returns:
        The loaded model object, or None if an error occurs.
    """
    try:
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Construct the correct path to the model file
        model_path = os.path.join(current_dir, 'model', 'best_model.pkl')
        
        # Ensure the file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at path: {model_path}")
        
        # Load the model using pickle
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Ensure the loaded object is a valid model
        if model is None:
            raise ValueError("The loaded model is None.")
        
        if not hasattr(model, 'predict'):
            raise ValueError("The loaded object is not a valid model with a 'predict' method.")
        
        print(f"Model loaded successfully. Model type: {type(model)}")
        return model

    except FileNotFoundError as fnf_error:
        print(f"Error: {fnf_error}")
        return None

    except ValueError as val_error:
        print(f"Error: {val_error}")
        return None

    except Exception as e:
        print(f"An unexpected error occurred while loading the model: {e}")
        return None


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'model', 'best_model.pkl')
    
    # Example input CSV file (replace with your actual data source)
    input_csv = "../data/data.csv"  # Replace with the path to your CSV file
    output_csv = "../data/preprocessed_data.csv"  # File to save preprocessed data

    try:
        # Load transaction data
        print("Loading transaction data...")
        df = pd.read_csv(input_csv)
        print("Transaction data loaded successfully.")

        # Preprocess the data
        print("Preprocessing transactions...")
        preprocessed_data = preprocess_transactions(df)
        print("Preprocessing complete.")

        # Save preprocessed data to CSV
        preprocessed_data.to_csv(output_csv, index=False)
        print(f"Preprocessed data saved to {output_csv}.")

        # Load the model
        print("Loading model...")
        print(f"Model path: {model_path}")

        model = load_model()

        if model is None:
            print("Model could not be loaded. Exiting.")
            sys.exit(1)

        # Make predictions
        print("Making predictions...")
        predictions = model.predict(preprocessed_data)
        print(f"Predictions: {predictions}")

   
