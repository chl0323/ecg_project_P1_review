# feature_analysis.py
# Feature analysis and statistical utilities for ECG deep learning project.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from mpl_toolkits.mplot3d import Axes3D

# Set English font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Create results directory
os.makedirs('analysis_results', exist_ok=True)

def convert_age_to_numeric(age_str):
    """Convert age string to numeric value"""
    if pd.isna(age_str):
        return np.nan
    if isinstance(age_str, (int, float)):
        return float(age_str)
    
    age_str = str(age_str).strip()
    if '>' in age_str:
        return 70.0  # For '>70'
    elif '<' in age_str:
        return 35.0  # For '<40'
    elif '-' in age_str:
        try:
            start, end = map(float, age_str.split('-'))
            return (start + end) / 2  # Return average of range
        except:
            return np.nan
    else:
        try:
            return float(age_str)
        except:
            return np.nan

def perform_pca_analysis(data, target, feature_names, n_components=None, 
                        save_dir='analysis_results/pca_analysis'):
    """
    Perform PCA analysis and visualization
    
    Args:
        data: Feature data
        target: Target variable
        feature_names: List of feature names
        n_components: Number of PCA components
        save_dir: Save directory
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Standardize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Perform PCA
    if n_components is None:
        n_components = min(data.shape[1], 10)  # Maximum 10 principal components
    
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data_scaled)
    
    # Calculate cumulative variance explained ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    print(f"\n=== PCA Analysis Results ===")
    print(f"Number of components: {n_components}")
    print(f"Cumulative variance explained ratio for the first {n_components} components: {cumulative_variance_ratio[-1]:.4f}")
    
    for i, ratio in enumerate(explained_variance_ratio):
        print(f"PC{i+1}: {ratio:.4f} ({ratio*100:.2f}%)")
    
    # 1. Variance explained ratio plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(1, n_components+1), explained_variance_ratio)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance Ratio')
    plt.xticks(range(1, n_components+1))
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, n_components+1), cumulative_variance_ratio, 'bo-')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
    plt.axhline(y=0.85, color='orange', linestyle='--', label='85% Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA Cumulative Variance Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/pca_variance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 2D PCA visualization
    plt.figure(figsize=(15, 5))
    
    # PC1 vs PC2
    plt.subplot(1, 3, 1)
    scatter = plt.scatter(data_pca[:, 0], data_pca[:, 1], c=target, 
                         cmap='viridis', alpha=0.6)
    plt.xlabel(f'PC1 ({explained_variance_ratio[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({explained_variance_ratio[1]:.2%} variance)')
    plt.title('PCA: PC1 vs PC2')
    plt.colorbar(scatter, label='Target')
    
    # PC2 vs PC3
    if n_components >= 3:
        plt.subplot(1, 3, 2)
        scatter = plt.scatter(data_pca[:, 1], data_pca[:, 2], c=target, 
                             cmap='viridis', alpha=0.6)
        plt.xlabel(f'PC2 ({explained_variance_ratio[1]:.2%} variance)')
        plt.ylabel(f'PC3 ({explained_variance_ratio[2]:.2%} variance)')
        plt.title('PCA: PC2 vs PC3')
        plt.colorbar(scatter, label='Target')
    
    # PC1 vs PC3
    if n_components >= 3:
        plt.subplot(1, 3, 3)
        scatter = plt.scatter(data_pca[:, 0], data_pca[:, 2], c=target, 
                             cmap='viridis', alpha=0.6)
        plt.xlabel(f'PC1 ({explained_variance_ratio[0]:.2%} variance)')
        plt.ylabel(f'PC3 ({explained_variance_ratio[2]:.2%} variance)')
        plt.title('PCA: PC1 vs PC3')
        plt.colorbar(scatter, label='Target')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/pca_2d_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 3D PCA visualization
    if n_components >= 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(data_pca[:, 0], data_pca[:, 1], data_pca[:, 2], 
                           c=target, cmap='viridis', alpha=0.6)
        ax.set_xlabel(f'PC1 ({explained_variance_ratio[0]:.2%})')
        ax.set_ylabel(f'PC2 ({explained_variance_ratio[1]:.2%})')
        ax.set_zlabel(f'PC3 ({explained_variance_ratio[2]:.2%})')
        ax.set_title('3D PCA Visualization')
        
        plt.colorbar(scatter, label='Target', shrink=0.8)
        plt.savefig(f'{save_dir}/pca_3d_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Interactive PCA visualization
    df_pca = pd.DataFrame(data_pca[:, :min(3, n_components)], 
                         columns=[f'PC{i+1}' for i in range(min(3, n_components))])
    df_pca['Target'] = target
    
    # 2D interactive plot
    fig = px.scatter(df_pca, x='PC1', y='PC2', color='Target',
                    title='Interactive PCA: PC1 vs PC2',
                    labels={'PC1': f'PC1 ({explained_variance_ratio[0]:.2%} variance)',
                           'PC2': f'PC2 ({explained_variance_ratio[1]:.2%} variance)'})
    fig.update_layout(template='plotly_white')
    pio.write_html(fig, f'{save_dir}/pca_interactive_2d.html')
    
    # 3D interactive plot
    if n_components >= 3:
        fig = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3', color='Target',
                           title='Interactive 3D PCA Visualization',
                           labels={'PC1': f'PC1 ({explained_variance_ratio[0]:.2%})',
                                  'PC2': f'PC2 ({explained_variance_ratio[1]:.2%})',
                                  'PC3': f'PC3 ({explained_variance_ratio[2]:.2%})'})
        fig.update_layout(template='plotly_white')
        pio.write_html(fig, f'{save_dir}/pca_interactive_3d.html')
    
    # 5. Principal component loading plot (Loading Plot)
    if len(feature_names) <= 20:  # Only plot when feature count is not too large
        plt.figure(figsize=(12, 8))
        
        loadings = pca.components_[:2].T  # Loadings of first two principal components
        
        plt.subplot(2, 2, 1)
        for i, feature in enumerate(feature_names):
            plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], 
                     head_width=0.02, head_length=0.02, fc='blue', ec='blue')
            plt.text(loadings[i, 0]*1.1, loadings[i, 1]*1.1, feature, 
                    fontsize=8, ha='center')
        plt.xlabel(f'PC1 ({explained_variance_ratio[0]:.2%})')
        plt.ylabel(f'PC2 ({explained_variance_ratio[1]:.2%})')
        plt.title('PCA Loading Plot: PC1 vs PC2')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # Loading matrix heatmap
        plt.subplot(2, 2, 2)
        sns.heatmap(pca.components_[:min(5, n_components)], 
                   xticklabels=feature_names, 
                   yticklabels=[f'PC{i+1}' for i in range(min(5, n_components))],
                   cmap='RdBu_r', center=0, annot=False)
        plt.title('PCA Components Heatmap')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/pca_loadings.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 6. Save PCA results
    pca_results = {
        'explained_variance_ratio': explained_variance_ratio.tolist(),
        'cumulative_variance_ratio': cumulative_variance_ratio.tolist(),
        'components': pca.components_.tolist(),
        'feature_names': feature_names,
        'n_components': n_components,
        'total_variance_explained': float(cumulative_variance_ratio[-1])
    }
    
    with open(f'{save_dir}/pca_results.json', 'w') as f:
        json.dump(pca_results, f, indent=4)
    
    # Save PCA transformed data
    np.save(f'{save_dir}/pca_transformed_data.npy', data_pca)
    
    return pca, data_pca, explained_variance_ratio

def perform_tsne_analysis(data, target, perplexity=30, max_iter=500,
                         save_dir='analysis_results/tsne_analysis'):
    """
    Perform t-SNE analysis
    
    Args:
        data: Feature data
        target: Target variable
        perplexity: t-SNE perplexity parameter
        max_iter: Maximum iterations
        save_dir: Save directory
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n=== t-SNE Analysis ===")
    print(f"Perplexity: {perplexity}, Iterations: {max_iter}")
    print("Starting t-SNE dimensionality reduction, this may take a few minutes...")
    
    # If data is too large, downsample first
    max_samples = 5000
    if len(data) > max_samples:
        print(f"Large data size ({len(data)} samples), downsampling to {max_samples} samples...")
        indices = np.random.choice(len(data), max_samples, replace=False)
        data = data[indices]
        target = target[indices]
    
    # Standardize data
    print("Standardizing data...")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Perform t-SNE
    print("Performing t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=max_iter, 
                random_state=42, verbose=1, n_jobs=-1)
    data_tsne = tsne.fit_transform(data_scaled)
    print("t-SNE dimensionality reduction completed!")
    
    # Visualize t-SNE results
    print("Generating visualizations...")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=target, 
                         cmap='viridis', alpha=0.6)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE Visualization')
    plt.colorbar(scatter, label='Target')
    
    # Density plot
    plt.subplot(1, 2, 2)
    for class_val in np.unique(target):
        mask = target == class_val
        plt.scatter(data_tsne[mask, 0], data_tsne[mask, 1], 
                   label=f'Class {class_val}', alpha=0.6)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE by Class')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/tsne_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Interactive t-SNE visualization
    print("Generating interactive visualization...")
    df_tsne = pd.DataFrame(data_tsne, columns=['t-SNE 1', 't-SNE 2'])
    df_tsne['Target'] = target
    
    fig = px.scatter(df_tsne, x='t-SNE 1', y='t-SNE 2', color='Target',
                    title='Interactive t-SNE Visualization')
    fig.update_layout(template='plotly_white')
    pio.write_html(fig, f'{save_dir}/tsne_interactive.html')
    
    # Save t-SNE results
    print("Saving results...")
    np.save(f'{save_dir}/tsne_transformed_data.npy', data_tsne)
    print("t-SNE analysis completed!")
    
    return data_tsne

# Read original data
print("Loading data...")
df = pd.read_excel('data_binary1.xlsx')  # Updated file name

# Convert age column to numeric
if 'anchor_age' in df.columns:
    df['anchor_age'] = df['anchor_age'].apply(convert_age_to_numeric)

# Print column names
print("\n=== Data Columns ===")
print(df.columns.tolist())

# Get all feature columns (excluding target column)
features = [col for col in df.columns if col != 'target']

# 1. Basic Statistics
print("\n=== Basic Statistics ===")
print(df[features].describe())

# 2. Outlier Detection (using 3 standard deviations as threshold)
print("\n=== Outlier Detection ===")
abnormal_stats = []
feature_stats = {}  # Store statistics for each feature

for feature in features:
    # Skip non-numeric columns
    if not np.issubdtype(df[feature].dtype, np.number):
        print(f"\nSkipping non-numeric feature: {feature}")
        continue
        
    mean = df[feature].mean()
    std = df[feature].std()
    threshold = 3 * std
    abnormal_count = len(df[abs(df[feature] - mean) > threshold])
    abnormal_ratio = abnormal_count/len(df)*100
    
    print(f"\n{feature} Outlier Statistics:")
    print(f"Mean: {mean:.4f}")
    print(f"Standard Deviation: {std:.4f}")
    print(f"Outlier Count (>3σ): {abnormal_count}")
    print(f"Outlier Ratio: {abnormal_ratio:.2f}%")
    
    abnormal_stats.append({
        'Feature': feature,
        'Mean': mean,
        'Std': std,
        'AbnormalCount': abnormal_count,
        'AbnormalRatio': abnormal_ratio
    })
    
    # Store feature statistics
    feature_stats[feature] = {
        'mean': float(mean),  # Convert to Python native type
        'std': float(std),
        'threshold': float(threshold),
        'abnormal_count': int(abnormal_count),
        'abnormal_ratio': float(abnormal_ratio)
    }

# Save feature statistics to JSON file
with open('analysis_results/feature_stats.json', 'w') as f:
    json.dump(feature_stats, f, indent=4)
print("\nFeature statistics saved to feature_stats.json")

# Convert abnormal statistics to DataFrame and sort
abnormal_df = pd.DataFrame(abnormal_stats)
abnormal_df = abnormal_df.sort_values('AbnormalRatio', ascending=False)
print("\n=== Outlier Ratio Ranking ===")
print(abnormal_df[['Feature', 'AbnormalCount', 'AbnormalRatio']])

# 3. Visualization
# 3.1 Outlier ratio bar plot
plt.figure(figsize=(15, 6))
sns.barplot(data=abnormal_df, x='Feature', y='AbnormalRatio')
plt.xticks(rotation=45, ha='right')
plt.title('Outlier Ratio by Feature')
plt.tight_layout()
plt.savefig('analysis_results/abnormal_ratios.png')
plt.close()

# 3.2 Feature distribution plots (4 features per row)
numeric_features = [f for f in features if np.issubdtype(df[f].dtype, np.number)]
n_features = len(numeric_features)
n_rows = (n_features + 3) // 4  # Round up
plt.figure(figsize=(20, 5*n_rows))

for i, feature in enumerate(numeric_features, 1):
    plt.subplot(n_rows, 4, i)
    sns.histplot(data=df, x=feature, bins=50)
    plt.title(f'{feature} Distribution')
    plt.axvline(df[feature].mean(), color='r', linestyle='--', label='Mean')
    plt.axvline(df[feature].mean() + 3*df[feature].std(), color='g', linestyle='--', label='3σ')
    plt.axvline(df[feature].mean() - 3*df[feature].std(), color='g', linestyle='--')
    plt.legend()

plt.tight_layout()
plt.savefig('analysis_results/feature_distributions.png')
plt.close()

# 4. Correlation Analysis
print("\n=== Correlation Analysis ===")
correlation = df[numeric_features].corr()
print(correlation)

# 5. Relationship with Target Variable
print("\n=== Relationship with Target Variable ===")
for feature in numeric_features:
    print(f"\n{feature} Statistics by Target Value:")
    print(df.groupby('target')[feature].describe())

# ===== New: PCA Analysis =====
# Prepare data for PCA analysis
numeric_data = df[numeric_features].dropna()
target_data = df.loc[numeric_data.index, 'target']

# Perform PCA analysis
print("\nStarting PCA analysis...")
pca_model, pca_transformed, explained_variance = perform_pca_analysis(
    numeric_data.values, target_data.values, numeric_features, n_components=10
)

# Perform t-SNE analysis
print("\nStarting t-SNE analysis...")
tsne_transformed = perform_tsne_analysis(
    numeric_data.values, target_data.values, perplexity=30, max_iter=500
)

# 6. Save Detailed Analysis Results
with open('analysis_results/feature_analysis.txt', 'w') as f:
    f.write("=== Feature Analysis Results ===\n\n")
    f.write("1. Data Columns:\n")
    f.write(str(df.columns.tolist()) + "\n\n")
    f.write("2. Basic Statistics:\n")
    f.write(str(df[numeric_features].describe()) + "\n\n")
    f.write("3. Outlier Statistics:\n")
    f.write(str(abnormal_df) + "\n\n")
    f.write("4. Correlation Analysis:\n")
    f.write(str(correlation) + "\n\n")
    f.write("5. Relationship with Target Variable:\n")
    for feature in numeric_features:
        f.write(f"\n{feature} Statistics by Target Value:\n")
        f.write(str(df.groupby('target')[feature].describe()) + "\n")

    # Add PCA analysis results
    f.write("\n\n6. PCA Analysis Results:\n")
    f.write(f"Number of components: {len(explained_variance)}\n")
    f.write(f"Total variance explained: {np.sum(explained_variance):.4f}\n")
    for i, ratio in enumerate(explained_variance):
        f.write(f"PC{i+1}: {ratio:.4f} ({ratio*100:.2f}%)\n")

print("Complete analysis completed! Results saved to analysis_results directory") 
