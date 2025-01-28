import unittest
import pandas as pd
import numpy as np
import sys, os
# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.feature_engg import FeatureEngineering

class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        # Sample DataFrame for testing
        self.df = pd.DataFrame({
            'TransactionId': [1, 2, 3, 4, 5],
            'CustomerId': ['A', 'A', 'B', 'B', 'C'],
            'Amount': [100, -50, 200, -100, 150],
            'TransactionStartTime': [
                '2024-01-01 10:00:00',
                '2024-01-02 11:00:00',
                '2024-01-03 12:00:00',
                '2024-01-04 13:00:00',
                '2024-01-05 14:00:00'
            ]
        })

    def test_create_aggregate_features(self):
        df_result = FeatureEngineering.create_aggregate_features(self.df)
        self.assertIn('Total_Transaction_Amount', df_result.columns)
        self.assertIn('Average_Transaction_Amount', df_result.columns)
        self.assertIn('Transaction_Count', df_result.columns)
        self.assertIn('Std_Transaction_Amount', df_result.columns)
        self.assertEqual(df_result['Transaction_Count'].iloc[0], 2)  # Customer A has 2 transactions

    def test_create_transaction_features(self):
        df_result = FeatureEngineering.create_transaction_features(self.df)
        self.assertIn('Net_Transaction_Amount', df_result.columns)
        self.assertIn('Debit_Count', df_result.columns)
        self.assertIn('Credit_Count', df_result.columns)
        self.assertIn('Debit_Credit_Ratio', df_result.columns)
        self.assertEqual(df_result['Debit_Count'].iloc[0], 1)  # Customer A has 1 debit
        self.assertEqual(df_result['Credit_Count'].iloc[0], 1)  # Customer A has 1 credit
        self.assertAlmostEqual(df_result['Debit_Credit_Ratio'].iloc[0], 0.5)  # Updated ratio for Customer A

    def test_extract_time_features(self):
        df_result = FeatureEngineering.extract_time_features(self.df)
        self.assertIn('Transaction_Hour', df_result.columns)
        self.assertIn('Transaction_Day', df_result.columns)
        self.assertIn('Transaction_Month', df_result.columns)
        self.assertIn('Transaction_Year', df_result.columns)
        self.assertEqual(df_result['Transaction_Hour'].iloc[0], 10)  # First transaction hour



    def test_handle_missing_values_mean(self):
        df_with_nans = pd.DataFrame({
            'TransactionId': [1, 2, 3],
            'Amount': [100, np.nan, 200]
        })
        df_result = FeatureEngineering.handle_missing_values(df_with_nans, strategy='mean')
        self.assertEqual(df_result['Amount'].isnull().sum(), 0)  # No missing values

   

if __name__ == '__main__':
    unittest.main()