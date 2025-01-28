import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pytz
import pandas as pd

class CreditScoreRFM:
    """
    A class to calculate Recency, Frequency, Monetary values and perform Weight of Evidence (WoE) binning,
    as well as merging these features into the original feature-engineered dataset.

    Attributes:
    -----------
    rfm_data : pd.DataFrame
        The dataset containing the transaction information for RFM calculation.

    Methods:
    --------
    calculate_rfm():
        Calculates Recency, Frequency, and Monetary values for each customer in the dataset.

    plot_pairplot():
        Plots a pair plot of the Recency, Frequency, and Monetary values.

    plot_heatmap():
        Plots a heatmap to visualize correlations between RFM variables.

    plot_histograms():
        Plots histograms for Recency, Frequency, and Monetary values.

    calculate_rfm_score(weight_recency=0.1, weight_frequency=0.5, weight_monetary=0.4):
        Calculates an RFM score based on Recency, Frequency, and Monetary values with adjustable weights.

    assign_label():
        Assigns users into "Good" and "Bad" categories based on the RFM score threshold.
    """

    def __init__(self, rfm_data):
        """
        Initializes the CreditScoreRFM class with the provided dataset.
        
        Parameters:
        -----------
        rfm_data : pd.DataFrame
            The input dataset containing transaction data.
        """
        self.rfm_data = rfm_data

    def calculate_rfm(self):
        """
        Calculates Recency, Frequency, and Monetary values for each customer.

        Returns:
            pandas.DataFrame: A DataFrame with additional columns for Recency, Frequency, Monetary, and RFM scores.
        """

        # Convert 'TransactionStartTime' to datetime and make it timezone-aware (UTC)
        self.rfm_data['TransactionStartTime'] = pd.to_datetime(self.rfm_data['TransactionStartTime'])

        # Set the end date to the current date and make it timezone-aware (UTC)
        end_date = pd.Timestamp.utcnow()

        # Calculate Recency, Frequency, and Monetary values
        self.rfm_data['Last_Access_Date'] = self.rfm_data.groupby('CustomerId')['TransactionStartTime'].transform('max')
        self.rfm_data['Recency'] = (end_date - self.rfm_data['Last_Access_Date']).dt.days
        self.rfm_data['Frequency'] = self.rfm_data.groupby('CustomerId')['TransactionId'].transform('count')

        if 'Amount' in self.rfm_data.columns:
            self.rfm_data['Monetary'] = self.rfm_data.groupby('CustomerId')['Amount'].transform('sum')
        else:
            # Handle missing Amount column (e.g., set to 1 for each transaction)
            self.rfm_data['Monetary'] = 1

        # Remove duplicates to create a summary DataFrame for scoring
        rfm_data = self.rfm_data[['CustomerId', 'Recency', 'Frequency', 'Monetary']].drop_duplicates()

        # # Calculate RFM scores
        # rfm_data = self.calculate_rfm_scores(rfm_data)

        # # Assign labels
        # rfm_data = self.assign_label(rfm_data)

        return rfm_data
    
    def calculate_rfm_scores(self, rfm_data):
        """
        Calculates RFM scores based on the Recency, Frequency, and Monetary values.

        Args:
            rfm_data (pandas.DataFrame): A DataFrame containing Recency, Frequency, and Monetary values.

        Returns:
            pandas.DataFrame: A DataFrame with additional columns for RFM scores.
        """
        
        # Quantile-based scoring
        rfm_data['r_quartile'] = pd.qcut(rfm_data['Recency'], 4, labels=['4', '3', '2', '1'])  # Lower recency is better
        rfm_data['f_quartile'] = pd.qcut(rfm_data['Frequency'], 4, labels=['1', '2', '3', '4'])  # Higher frequency is better
        rfm_data['m_quartile'] = pd.qcut(rfm_data['Monetary'], 4, labels=['1', '2', '3', '4'])  # Higher monetary is better

        # Calculate overall RFM Score
        rfm_data['RFM_Score'] = (rfm_data['r_quartile'].astype(int) * 0.1 +
                                  rfm_data['f_quartile'].astype(int) * 0.45 +
                                  rfm_data['m_quartile'].astype(int) * 0.45)
        high_threshold = rfm_data['RFM_Score'].quantile(0.75)  # Change to .75 to include moderate users
        low_threshold = rfm_data['RFM_Score'].quantile(0.5)  # Change to .25 to include moderate users
        rfm_data['Risk_Label'] = rfm_data['RFM_Score'].apply(lambda x: 'Good' if x >= low_threshold else 'Bad')
        return rfm_data
    
    def assign_label(self, rfm_data):
        """ 
        Assign 'Good' or 'Bad' based on the RFM Score threshold (e.g., median).
        
        Args:
            rfm_data (pandas.DataFrame): A DataFrame with RFM scores.
        
        Returns:
            pandas.DataFrame: Updated DataFrame with Risk_Label column.
        """
        high_threshold = rfm_data['RFM_Score'].quantile(0.75)  # Change to .75 to include moderate users
        low_threshold = rfm_data['RFM_Score'].quantile(0.5)  # Change to .25 to include moderate users
        rfm_data['Risk_Label'] = rfm_data['RFM_Score'].apply(lambda x: 'Good' if x >= low_threshold else 'Bad')
        return rfm_data
    
   
    def calculate_counts(self, data):
        # Ensure consistent bin indexing
        all_bins = sorted(data['RFM_bin'].dropna().unique())
        grouped_data = data.groupby('RFM_bin')
    
        # Calculate counts with reindexing
        good_count = grouped_data['Risk_Label'].apply(lambda x: (x == 'Good').sum()).reindex(all_bins, fill_value=0)
        bad_count = grouped_data['Risk_Label'].apply(lambda x: (x == 'Bad').sum()).reindex(all_bins, fill_value=0)
    
        return good_count, bad_count

    #def calculate_woe(self, good_count, bad_count):
    #    total_good = good_count.sum()
    #    total_bad = bad_count.sum()
#
    #    # Add epsilon (small value) to avoid log(0) or division by zero
    #    epsilon = 1e-10
    #    
    #    good_rate = good_count / (total_good + epsilon)  # Avoid division by zero
    #    bad_rate = bad_count / (total_bad + epsilon)     # Avoid division by zero
#
    #    # Calculate WoE and handle cases where bad_count is zero
    #    woe = np.log((good_rate + epsilon) / (bad_rate + epsilon))  # Add epsilon to rates
    #    # Calculate IV for each bin
    #    iv = ((good_rate - bad_rate) * woe).sum()  # Sum over all bins to get the total IV
    #    
    #    return woe, iv
    def calculate_woe(self, good_count, bad_count):
        # Avoid division by zero
        total_good = good_count.sum()
        total_bad = bad_count.sum()
    
        # Add a small value (1e-9) to avoid log errors
        woe_values = np.log(
            ((good_count + 1e-9) / total_good) / ((bad_count + 1e-9) / total_bad)
        )
        return woe_values

    def plot_pairplot(self):
        """
        Creates a pair plot to visualize relationships between Recency, Frequency, and Monetary.
        """
        sns.pairplot(self.rfm_data[['Recency', 'Frequency', 'Monetary']])
        plt.suptitle('Pair Plot of rfm Variables', y=1.02)
        plt.show()

    def plot_heatmap(self):
        """
        Creates a heatmap to visualize correlations between rfm variables.
        """
        corr = self.rfm_data[['Recency', 'Frequency', 'Monetary']].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix of rfm Variables')
        plt.show()

    def plot_histograms(self):
        """
        Plots histograms for Recency, Frequency, and Monetary.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        self.rfm_data['Recency'].hist(bins=20, ax=axes[0])
        axes[0].set_title('Recency Distribution')
        
        self.rfm_data['Frequency'].hist(bins=20, ax=axes[1])
        axes[1].set_title('Frequency Distribution')
        
        self.rfm_data['Monetary'].hist(bins=20, ax=axes[2])
        axes[2].set_title('Monetary Distribution')

        plt.tight_layout()
        plt.show()