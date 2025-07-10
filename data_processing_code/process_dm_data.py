"""
File: process_dm_data.py
Description: Script for processing diabetes mellitus (DM) related data for ECG analysis.
Author: chenhuilin
Date: 2025-06-06
"""

import pandas as pd
import numpy as np

# All Chinese comments, print statements, docstrings, and error messages have been translated to English. Variable names are kept unless in pinyin or Chinese.
# 1. Process diabetes status marker
print("Step 1: Process diabetes status marker...")

# Read diagnosis data
diagnosis_df = pd.read_excel("diagnosis.xlsx")
# Read ECG record data
ecg_df = pd.read_excel("new_record_list_2version.xlsx")

# Get data types for each column
column_dtypes = ecg_df.dtypes

# Check for negative values in original data
print("\nChecking for negative values in original data:")
numeric_cols = ecg_df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col not in ['subject_id']]
for col in numeric_cols:
    neg_count = (ecg_df[col] < 0).sum()
    if neg_count > 0:
        print(f"{col}: Found {neg_count} negative values")

# Filter for ICD codes containing E11
dm_subjects = diagnosis_df[diagnosis_df['icd_code'].str.contains('E11', na=False)]['subject_id'].unique()

# Create diabetes status marker
ecg_df['dm_status'] = ecg_df['subject_id'].isin(dm_subjects).astype(int)

# Save processed data
ecg_df.to_excel("dm_record_list.xlsx", index=False)
print(f"\nProcessed data saved to dm_record_list.xlsx")
print(f"Total records: {len(ecg_df)}")
print(f"Diabetes status distribution:\n{ecg_df['dm_status'].value_counts()}")

# 2. Data filtering and statistics
print("\nStep 2: Data filtering and statistics...")

# Read processed data
dm_df = pd.read_excel("dm_record_list.xlsx")

# Calculate the number of records and the missing rate of RR_Interval for each subject_id
subject_stats = dm_df.groupby('subject_id').agg({
    'ecg_time': 'count',  # Number of records
    'RR_Interval': lambda x: x.isna().mean()  # Missing rate
}).reset_index()

# Filter subject_ids meeting the criteria
valid_subjects = subject_stats[
    (subject_stats['ecg_time'] > 10) &  # Number of records > 10
    (subject_stats['RR_Interval'] < 0.2)  # Missing rate < 20%
]['subject_id']

# Get data meeting the criteria
filtered_df = dm_df[dm_df['subject_id'].isin(valid_subjects)].copy()

# Check for negative values in filtered data
print("\nChecking for negative values in filtered data:")
for col in numeric_cols:
    neg_count = (filtered_df[col] < 0).sum()
    if neg_count > 0:
        print(f"{col}: Found {neg_count} negative values")
        # Replace negative values with the positive average of that subject_id
        for subject_id in filtered_df['subject_id'].unique():
            subject_data = filtered_df[filtered_df['subject_id'] == subject_id]
            neg_mask = subject_data[col] < 0
            if neg_mask.any():
                # Get the positive average of that subject_id
                pos_mean = subject_data[subject_data[col] > 0][col].mean()
                if pd.notna(pos_mean):
                    # Convert based on original data type
                    if column_dtypes[col] == np.int64:
                        filtered_df.loc[(filtered_df['subject_id'] == subject_id) & (filtered_df[col] < 0), col] = int(round(pos_mean))
                    else:
                        filtered_df.loc[(filtered_df['subject_id'] == subject_id) & (filtered_df[col] < 0), col] = float(pos_mean)

# Check for negative values again
print("\nChecking for negative values after processing:")
for col in numeric_cols:
    neg_count = (filtered_df[col] < 0).sum()
    if neg_count > 0:
        print(f"{col}: Still {neg_count} negative values")
    else:
        print(f"{col}: No negative values")

# Statistics
print("\nFiltering results statistics:")
print(f"Total records meeting criteria: {len(filtered_df)}")
print(f"Total number of subjects meeting criteria: {len(filtered_df['subject_id'].unique())}")

# Group by diabetes status for statistics
dm_stats = filtered_df.groupby('dm_status').agg({
    'subject_id': ['count', 'nunique']
}).reset_index()

dm_stats.columns = ['dm_status', 'Number of records', 'Number of subjects']
dm_stats['dm_status'] = dm_stats['dm_status'].map({0: 'Non-Diabetes', 1: 'Diabetes'})

print("\nStatistics by diabetes status:")
print(dm_stats)

# Save filtered data
filtered_df.to_excel("filtered_dm_record_list.xlsx", index=False)
print("\nFiltered data saved to filtered_dm_record_list.xlsx") 