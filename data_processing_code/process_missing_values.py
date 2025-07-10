"""
File: process_missing_values.py
Description: Script for handling missing values in ECG data.
Author: chenhuilin
Date: 2025-06-06
"""

import pandas as pd
import numpy as np

# All Chinese comments, print statements, docstrings, and error messages have been translated to English. Variable names are kept unless in pinyin or Chinese.
# Read filtered data
print("Reading data...")
df = pd.read_excel("filtered_dm_record_list.xlsx")

# Get all numeric columns (except subject_id, dm_status)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col not in ['subject_id', 'dm_status']]

# Get data type of each column
column_dtypes = df.dtypes

# Calculate statistics for each group
print("Calculating statistics...")
# Calculate statistics for non-diabetic group (dm_status=0)
dm0_stats = df[df['dm_status'] == 0].agg({
    'gender': lambda x: x.mode()[0],  # Mode
    **{col: 'median' for col in numeric_cols}  # Median
})

# Calculate statistics for diabetic group (dm_status=1)
dm1_stats = df[df['dm_status'] == 1].agg({
    'gender': lambda x: x.mode()[0],  # Mode
    **{col: 'median' for col in numeric_cols}  # Median
})

# Create a copy for processing
df_processed = df.copy()

# Handle missing values and zeros
print("Handling missing values and zeros...")
for index, row in df_processed.iterrows():
    subject_id = row['subject_id']
    dm_status = row['dm_status']
    
    # Get all records for this subject_id
    subject_data = df[df['subject_id'] == subject_id]
    
    # Iterate through each column
    for col in df_processed.columns:
        # Check if it's a missing value or zero
        if pd.isna(row[col]) or (col in numeric_cols and row[col] == 0):
            # Try to use a valid value for this subject_id
            if col == 'gender':
                # For gender, use the mode
                valid_values = subject_data[col].dropna()
                if not valid_values.empty:
                    subject_value = valid_values.mode()[0]
                else:
                    subject_value = None
            else:
                # For other columns, use the non-zero mean
                valid_values = subject_data[col][subject_data[col] != 0].dropna()
                if not valid_values.empty:
                    subject_value = valid_values.mean()
                else:
                    subject_value = None
            
            # If the subject_id has no valid value, use the statistics for the corresponding dm_status
            if pd.isna(subject_value) or subject_value == 0:
                if dm_status == 0:
                    df_processed.at[index, col] = dm0_stats[col]
                else:
                    df_processed.at[index, col] = dm1_stats[col]
            else:
                # Convert based on original data type
                if col in numeric_cols:
                    if column_dtypes[col] == np.int64:
                        df_processed.at[index, col] = int(round(subject_value))
                    else:
                        df_processed.at[index, col] = float(subject_value)
                else:
                    df_processed.at[index, col] = subject_value

# Save processed data
print("Saving processed data...")
df_processed.to_excel("processed_dm_data.xlsx", index=False)

# Print processing statistics
print("\nProcessing statistics:")
print(f"Original data missing and zero values:")
for col in df.columns:
    missing_count = df[col].isna().sum()
    zero_count = ((df[col] == 0) & (col in numeric_cols)).sum()
    print(f"{col}: Missing={missing_count}, Zero={zero_count}")

print(f"\nProcessed data missing and zero values:")
for col in df_processed.columns:
    missing_count = df_processed[col].isna().sum()
    zero_count = ((df_processed[col] == 0) & (col in numeric_cols)).sum()
    print(f"{col}: Missing={missing_count}, Zero={zero_count}")

# Save processing statistics
stats_df = pd.DataFrame({
    'Field': numeric_cols + ['gender'],
    'Median/Mode (Non-Diabetic)': [dm0_stats[col] for col in numeric_cols + ['gender']],
    'Median/Mode (Diabetic)': [dm1_stats[col] for col in numeric_cols + ['gender']]
})
stats_df.to_excel("missing_value_stats.xlsx", index=False)
print("\nStatistics saved to missing_value_stats.xlsx") 