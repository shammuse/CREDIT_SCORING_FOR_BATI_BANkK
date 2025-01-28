#Credit EDA visualize
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Define the CreditRiskEDA class
class CreditRiskEDAVisualize:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the EDA class with the DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            The dataset to be analyzed.
        """
        self.df = df
    def plot_numerical_distribution(self, cols):
        """
        Function to plot multiple histograms in a grid layout.

        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame containing the dataset.
        cols : list
            List of numeric columns to plot.
        n_rows : int
            Number of rows in the grid.
        n_cols : int
            Number of columns in the grid.
        """

        # Select numeric columns
        n_cols = len(cols)

        # Automatically determine grid size (square root method)
        n_rows = math.ceil(n_cols**0.5)
        n_cols = math.ceil(n_cols / n_rows)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 6))
        axes = axes.flatten()

        for i, col in enumerate(cols):
            sns.histplot(self.df[col], bins=15, kde=True, color='skyblue', edgecolor='black', ax=axes[i])
            axes[i].set_title(f'Distribution of {col}', fontsize=14)
            axes[i].set_xlabel(col, fontsize=12)
            axes[i].set_ylabel('Frequency', fontsize=12)
            axes[i].axvline(self.df[col].mean(), color='red', linestyle='dashed', linewidth=1)
            axes[i].axvline(self.df[col].median(), color='green', linestyle='dashed', linewidth=1)
            axes[i].legend({'Mean': self.df[col].mean(), 'Median': self.df[col].median()})

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()
        
    # Function to plot skewness for each numerical feature
    def plot_skewness(self):
        df = self.df.select_dtypes(include='number')
        skewness = df.skew().sort_values(ascending=False)
        
        plt.figure(figsize=(10, 4))
        sns.barplot(x=skewness.index, y=skewness.values, hue=skewness.index, legend=False, palette="coolwarm")
        plt.title("Skewness of Numerical Features", fontsize=16)
        plt.xlabel("Features", fontsize=12)
        plt.ylabel("Skewness", fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        
    def plot_categorical_distribution(self):
        """
        Function to plot the distribution of categorical features in a DataFrame and 
        display the count values on top of each bar.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame containing the dataset to be analyzed.
        """
        # Select categorical columns
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        cols_with_few_categories = [col for col in categorical_cols if self.df[col].nunique() <= 10]

        # Set up the grid for subplots
        num_cols = len(cols_with_few_categories)
        num_rows = (num_cols + 1) // 2  # Automatically determine the grid size
        
        fig, axes = plt.subplots(num_rows, 2, figsize=(15, num_rows * 5))
        axes = axes.flatten()

        for i, col in enumerate(cols_with_few_categories):
            ax = sns.countplot(data=self.df, x=col, ax=axes[i], hue=col, legend=False, palette="Set2")
            axes[i].set_title(f'Distribution of {col}', fontsize=14)
            axes[i].tick_params(axis='x', rotation=90)
            axes[i].set_xlabel('')
            axes[i].set_ylabel('Frequency')
            # Add count labels to the top of the bars
            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}', 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='baseline', fontsize=12, color='black', 
                            xytext=(0, 5), textcoords='offset points')
            
        # Remove any empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.show()

    def correlation_analysis(self):
        """Generate and visualize the correlation matrix."""
        corr_matrix = self.df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Matrix', fontsize=16)
        plt.show()
    
    def check_missing_values(self):
        """Check for missing values and visualize the missing data pattern."""
        missing_values = self.df.isnull().sum()
        print("\nMissing Values in Each Column:")
        print(missing_values[missing_values > 0])
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.df.isnull(), cbar=False, cmap='viridis')
        plt.title('Missing Values Heatmap', fontsize=14)
        plt.show()

    def detect_outliers(self, cols):
        """
        Function to plot boxplots for numerical features to detect outliers.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame containing the dataset to be analyzed.
        numerical_cols : list
            List of numerical columns to plot.
        """
        # num_cols = self.df.select_dtypes(include=[np.number]).columns
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(cols, 1):
            plt.subplot(3, 3, i)
            sns.boxplot(y=self.df[col], color='orange')
            plt.title(f'Boxplot of {col}', fontsize=12)
        plt.tight_layout()
        plt.show()