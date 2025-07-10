"""
File: process_additional_features.py
Description: Script for processing and engineering additional ECG features.
Author: chenhuilin
Date: 2025-06-06
"""

import pandas as pd
import numpy as np

# 1. Load data
print("Step 1: Load data...")
df = pd.read_excel('/Users/pursuing/Downloads/project/second_data_processing_forms/processed_dm_data.xlsx')

# 2. Convert target variable
print("\nStep 2: Convert target variable...")
df['target'] = df['dm_status'].apply(lambda x: 1 if x == 2 else 0)

# 3. Feature selection
print("\nStep 3: Select features...")
# Basic features
base_features = ['RR_Interval', 'PR_Interval', 'QRS_Complex', 'QT_Interval', 'QTc_Interval',
                'P_Wave_Peak', 'R_Wave_Peak', 'T_Wave_Peak', 'HRV_SDNN', 'QTc_variability', 
                'ecg_sequence', 'anchor_age', 'gender']

# 4. Create new DataFrame
print("\nStep 4: Create new DataFrame...")
new_df = df[base_features].copy()
new_df['target'] = df['target']
new_df['subject_id'] = df['subject_id']

# 5. Gender encoding
print("\nStep 5: Process gender encoding...")
if new_df['gender'].dtype == object:
    new_df['gender'] = new_df['gender'].map({'M': 0, 'F': 1})

# 6. Add new features
print("\nStep 6: Add new features...")

# 6.1 Calculate heart rate related features
new_df['heart_rate'] = 60000 / new_df['RR_Interval']  # Heart rate (bpm)

# 6.2 Calculate QTc related ratios
new_df['QT_RR_ratio'] = new_df['QT_Interval'] / new_df['RR_Interval']
new_df['QTc_RR_ratio'] = new_df['QTc_Interval'] / new_df['RR_Interval']

# 6.3 Calculate peak ratios
new_df['R_P_ratio'] = new_df['R_Wave_Peak'] / new_df['P_Wave_Peak']
new_df['T_R_ratio'] = new_df['T_Wave_Peak'] / new_df['R_Wave_Peak']

# 6.4 Calculate age groups
new_df['age_group'] = pd.cut(new_df['anchor_age'],
                            bins=[0, 40, 50, 60, 70, 100],
                            labels=['<40', '40-50', '50-60', '60-70', '>70'])

# 6.5 Calculate heart rate variability metrics
new_df['HRV_CV'] = new_df['HRV_SDNN'] / new_df['RR_Interval'].mean()  # Coefficient of variation

# 6.6 Calculate QT dispersion
new_df['QT_dispersion'] = new_df['QT_Interval'].max() - new_df['QT_Interval'].min()

# 7. Handle missing values
print("\nStep 7: Handle missing values...")
# For numeric features, fill with median
numeric_cols = new_df.select_dtypes(include=[np.number]).columns
new_df[numeric_cols] = new_df[numeric_cols].fillna(new_df[numeric_cols].median())

# 8. Data standardization
print("\nStep 8: Data standardization...")
# Standardize numeric features
numeric_cols = new_df.select_dtypes(include=[np.number]).columns
numeric_cols = [col for col in numeric_cols if col not in ['target', 'subject_id']]
new_df[numeric_cols] = (new_df[numeric_cols] - new_df[numeric_cols].mean()) / new_df[numeric_cols].std()

# 9. Save processed data
print("\nStep 9: Save data...")
new_df.to_excel('data_binary2.xlsx', index=False)

# 10. Output data statistics
print("\nData statistics:")
print(f"Total samples: {len(new_df)}")
print(f"Positive samples: {sum(new_df['target'] == 1)}")
print(f"Negative samples: {sum(new_df['target'] == 0)}")
print(f"Positive sample ratio: {sum(new_df['target'] == 1)/len(new_df):.2%}")

# 11. Display feature list
print("\nFeature list:")
print(new_df.columns.tolist())

# 12. Display data preview
print("\nData preview:")
print(new_df.head()) 