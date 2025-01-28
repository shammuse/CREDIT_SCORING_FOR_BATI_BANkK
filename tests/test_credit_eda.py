import unittest
import pandas as pd
import numpy as np
import sys, os
# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.credit_eda_analysis import CreditRiskEDA
from scripts.credit_eda_visualize import CreditRiskEDAVisualize
import matplotlib.pyplot as plt

class TestCreditRiskEDA(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Creating a sample dataset for testing
        data = {
            'Age': [25, 35, 45, 32, 22],
            'Income': [50000, 60000, 80000, 70000, 65000],
            'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
            'SpendingScore': [60, 40, 70, 50, 90],
            'TransactionCount': [10, 20, 15, 5, 12]
        }
        cls.df = pd.DataFrame(data)
        cls.eda = CreditRiskEDA(cls.df)
        cls.eda_vis = CreditRiskEDAVisualize(cls.eda)

    def test_data_overview(self):
        # Test the data_overview method
        try:
            self.eda.data_overview()
            self.assertTrue(True)  # If no error occurs, the test passes
        except Exception as e:
            self.fail(f"data_overview method raised an exception: {e}")

    def test_summary_statistics(self):
        # Test the summary_statistics method
        summary_stats = self.eda.summary_statistics()
        self.assertIsInstance(summary_stats, pd.DataFrame)
        self.assertTrue('median' in summary_stats.columns)
        self.assertTrue('skewness' in summary_stats.columns)

    def test_plot_numerical_distribution(self):
        # Test the plot_numerical_distribution method
        try:
            self.eda_vis.plot_numerical_distribution(['Age', 'Income', 'SpendingScore', 'TransactionCount'])
            self.assertTrue(True)  # If no error occurs, the test passes
        except Exception as e:
            self.fail(f"plot_numerical_distribution method raised an exception: {e}")

    def test_plot_skewness(self):
        # Test the plot_skewness method
        try:
            self.eda_vis.plot_skewness()
            self.assertTrue(True)  # If no error occurs, the test passes
        except Exception as e:
            self.fail(f"plot_skewness method raised an exception: {e}")

    def test_plot_categorical_distribution(self):
        # Test the plot_categorical_distribution method
        try:
            self.eda_vis.plot_categorical_distribution()
            self.assertTrue(True)  # If no error occurs, the test passes
        except Exception as e:
            self.fail(f"plot_categorical_distribution method raised an exception: {e}")

    def test_correlation_analysis(self):
        # Test the correlation_analysis method
        # Ensure only numeric columns are included
        numeric_df = self.df.select_dtypes(include='number')
        if numeric_df.empty:
            self.skipTest("No numeric columns available for correlation analysis.")
        try:
            self.eda_vis.correlation_analysis()
            self.assertTrue(True)  # If no error occurs, the test passes
        except Exception as e:
            self.fail(f"correlation_analysis method raised an exception: {e}")

   #def test_check_missing_values(self):
   #    # Test the check_missing_values method
   #    try:
   #        self.eda_vis.check_missing_values()
   #        self.assertTrue(True)  # If no error occurs, the test passes
   #    except Exception as e:
   #        self.fail(f"check_missing_values method raised an exception: {e}")

    def test_detect_outliers(self):
        # Test the detect_outliers method
        try:
            self.eda_vis.detect_outliers(['Age', 'Income', 'SpendingScore', 'TransactionCount'])
            self.assertTrue(True)  # If no error occurs, the test passes
        except Exception as e:
            self.fail(f"detect_outliers method raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()