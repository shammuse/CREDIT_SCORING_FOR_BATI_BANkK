# Data Analysis Report on Credit Transactions

## Overview
This project provides a comprehensive analysis of credit transaction data. The primary objective is to explore the dataset, assess data quality, and prepare the data for further analysis, identifying trends, patterns, and anomalies within the data.

## Table of Contents
- [Introduction](#introduction)
- [Data Overview](#data-overview)
- [Data Quality Assessment](#data-quality-assessment)
- [Summary Statistics](#summary-statistics)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Cleaning Process](#data-cleaning-process)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Introduction
This report analyzes credit transaction data obtained from [source]. It aims to understand the dataset's structure and integrity while preparing it for further analysis.

## Data Overview
The dataset consists of 95,662 entries and 18 columns, containing various attributes of credit transactions.

### Key Attributes
- **TransactionId**: Unique identifier for each transaction.
- **AccountId**: Identifier for the customer’s account.
- **Amount**: The monetary value associated with each transaction.
- **FraudResult**: Indicates whether a transaction is fraudulent (1) or not (0).

## Data Quality Assessment
### Missing Values
Certain columns contain missing values, including:
- AccountId: 2 missing values
- CountryCode: 2 missing values
- ProviderId: 2 missing values
- Value: 3 missing values
- TransactionStartTime: 1 missing value
- PricingStrategy: 3 missing values
- Unnamed Columns: Significant missing data

### Duplicates
Duplicates were identified and removed, ensuring data integrity.

## Summary Statistics
Summary statistics provide insight into the distribution of numerical features, such as:
- **Amount**: Mean: $6,718.07, Min: -$1,000,000.00, Max: $9,880,000.00
- **Value**: Mean: $9,900.64, Min: $2.00, Max: $9,880,000.00

## Exploratory Data Analysis
### Distribution of Numerical Features
Histograms and KDE plots reveal a right-skewed distribution for the Amount feature.

### Distribution of Categorical Features
A count plot shows that only a small percentage of transactions are fraudulent (approximately 0.2%).

### Correlation Analysis
Weak correlations among numerical features were observed, with no significant predictors of fraud identified.

## Data Cleaning Process
### Handling Missing Values
Missing numeric values were filled with the median. Rows with missing categorical values were removed.

### Removing Duplicates
Duplicates were removed, resulting in a final dataset of 54,942 entries.

### Outlier Treatment
Outliers were removed based on the 1.5 * IQR rule.

## Conclusion
The data cleaning process resulted in a refined dataset of 54,942 entries, ready for further exploration or predictive modeling.

## Future Work
Further analysis could involve:
- Predictive modeling to identify potential fraudulent transactions.
- Time-series analysis to observe trends over time.

## Installation
To run this project, ensure you have the following libraries installed:
```bash
pip install pandas matplotlib seaborn python-docx
