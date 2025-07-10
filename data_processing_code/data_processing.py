"""
File: data_processing2.py
Description: Data preprocessing pipeline for ECG data (cleaning, normalization, etc.).
Author: chenhuilin
Date: 2025-06-06
"""

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import os
from scipy import signal
from scipy.signal import butter, filtfilt

def butter_bandpass(lowcut, highcut, fs, order=5):
    """Design a bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply a bandpass filter."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)

def normalize_signal(signal_data):
    """Normalize a signal."""
    return (signal_data - np.mean(signal_data)) / np.std(signal_data)

def remove_baseline_wander(signal_data, fs):
    """Remove baseline wander."""
    # Use a high-pass filter to remove baseline wander
    b, a = butter(4, 0.5/(fs/2), btype='high')
    return filtfilt(b, a, signal_data)

def preprocess_ecg_signal(signal_data, fs=250):
    """Complete ECG signal preprocessing pipeline."""
    # 1. Remove baseline wander
    signal_filtered = remove_baseline_wander(signal_data, fs)
    
    # 2. Apply bandpass filter (0.5-40 Hz)
    signal_filtered = apply_bandpass_filter(signal_filtered, 0.5, 40, fs)
    
    # 3. Normalize
    signal_normalized = normalize_signal(signal_filtered)
    
    return signal_normalized

# Define basic feature columns
def get_base_feature_cols():
    return [
    'RR_Interval', 'PR_Interval', 'QRS_Complex', 'QT_Interval', 'QTc_Interval',
    'P_Wave_Peak', 'R_Wave_Peak', 'T_Wave_Peak', 'HRV_SDNN', 'QTc_variability',
    'anchor_age', 'gender', 'ecg_sequence', 'heart_rate']

# Define features to calculate
def calculate_additional_features(df):
    """Calculate additional features."""
    print("[Feature Calculation] Starting to calculate additional features...")
    
    # Calculate QTc related ratios
    df['QTc_RR_ratio'] = df['QTc_Interval'] / df['RR_Interval']
    df['QT_RR_ratio'] = df['QT_Interval'] / df['RR_Interval']
    
    # Calculate peak ratios
    df['P_R_ratio'] = df['P_Wave_Peak'] / df['R_Wave_Peak']
    df['T_R_ratio'] = df['T_Wave_Peak'] / df['R_Wave_Peak']
    
    # Calculate heart rate variability metrics
    for subject_id in df['subject_id'].unique():
        subject_data = df[df['subject_id'] == subject_id]
        rr_intervals = subject_data['RR_Interval'].dropna()
        if len(rr_intervals) > 1:
            # RMSSD
            df.loc[df['subject_id'] == subject_id, 'HRV_RMSSD'] = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
            # pNN50
            nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 50)
            df.loc[df['subject_id'] == subject_id, 'HRV_pNN50'] = (nn50 / (len(rr_intervals) - 1)) * 100
    
    print("[Feature Calculation] Completed!")
    return df

# Get all feature columns
def get_all_feature_cols():
    base_cols = get_base_feature_cols()
    additional_cols = ['QTc_RR_ratio', 'QT_RR_ratio', 'P_R_ratio', 'T_R_ratio', 'HRV_RMSSD', 'HRV_pNN50']
    return base_cols + additional_cols

feature_cols = get_all_feature_cols()

sequence_length = 10

def redistribute_subject_ids(df, sequence_length=10):
    """
    Redistribute subject_id, grouping ECG data for each subject_id into sequences of length sequence_length.
    If there is not enough data for a sequence, it is discarded.
    """
    print("[Data Redistribution] Starting to redistribute subject_id...")
    new_df = pd.DataFrame()
    new_subject_id = 19998592  # Start from 19998592
    
    for subject_id in df['subject_id'].unique():
        subject_data = df[df['subject_id'] == subject_id].copy()
        n_sequences = len(subject_data) // sequence_length
        
        # Process complete sequences
        for i in range(n_sequences):
            start_idx = i * sequence_length
            end_idx = (i + 1) * sequence_length
            sequence_data = subject_data.iloc[start_idx:end_idx].copy()
            sequence_data['subject_id'] = new_subject_id
            sequence_data['ecg_sequence'] = range(1, sequence_length + 1)
            new_df = pd.concat([new_df, sequence_data])
            new_subject_id += 1
    
    print(f"[Data Redistribution] Completed! Original subject_id count: {len(df['subject_id'].unique())}, New subject_id count: {new_subject_id-19998592}")
    return new_df

def prepare_sequence_data_by_subject(data, sequence_length=10):
    X_sequences, y_sequences = [], []
    for subject_id in data['subject_id'].unique():
        subject_data = data[data['subject_id'] == subject_id]
        subject_features = subject_data[feature_cols].values
        subject_labels = subject_data['target'].values
        # Since it's already grouped by sequence_length, use the entire sequence directly
        X_sequences.append(subject_features)
        y_sequences.append(subject_labels[0])  # Use the first label of the sequence
    return np.array(X_sequences), np.array(y_sequences)

def load_and_preprocess_data(test_size=0.2, val_size=0.1, random_state=42):
    """
    Load data and perform preprocessing, including sequence generation, data balancing, and dataset splitting.
    """
    print("[Data Preprocessing] Starting...")
    df = pd.read_excel('data_binary1.xlsx')
    
    # Print data information
    print("\n[Data Information]")
    print("Column names:", df.columns.tolist())
    print("Data shape:", df.shape)
    print("Data types:\n", df.dtypes)
    
    # Check if subject_id column exists
    if 'subject_id' not in df.columns:
        print("\n[Error] subject_id column not found!")
        print("Available column names:", df.columns.tolist())
        return None
    
    # Calculate additional features
    df = calculate_additional_features(df)
    
    # Redistribute subject_id
    df = redistribute_subject_ids(df, sequence_length)
    
    # Print data information after redistribution
    print("\n[Data Information after Redistribution]")
    print("Column names:", df.columns.tolist())
    print("Data shape:", df.shape)
    print("Data types:\n", df.dtypes)
    
    X_seq, y_seq = prepare_sequence_data_by_subject(df, sequence_length)
    X_seq_2d = X_seq.reshape(X_seq.shape[0], -1)
    
    smote = SMOTE(random_state=random_state)
    X_balanced, y_balanced = smote.fit_resample(X_seq_2d, y_seq)
    
    class_weights = {0: 1, 1: len(y_seq[y_seq == 0]) / len(y_seq[y_seq == 1])}
    
    os.makedirs('processed_data', exist_ok=True)
    np.save('processed_data/X_balanced.npy', X_balanced)
    np.save('processed_data/y_balanced.npy', y_balanced)
    
    # Split dataset
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_balanced, y_balanced, test_size=test_size, random_state=random_state, stratify=y_balanced)
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp)
    
    np.save('processed_data/X_train_smote.npy', X_train)
    np.save('processed_data/y_train_smote.npy', y_train)
    np.save('processed_data/X_val_smote.npy', X_val)
    np.save('processed_data/y_val_smote.npy', y_val)
    np.save('processed_data/X_test_smote.npy', X_test)
    np.save('processed_data/y_test_smote.npy', y_test)
    
    print(f"[Data Preprocessing] Completed! Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
    
    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test),
        'full': (X_balanced, y_balanced),
        'class_weights': class_weights
    }

if __name__ == "__main__":
    data_dict = load_and_preprocess_data(
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    print("All datasets have been saved to the 'processed_data' directory.")