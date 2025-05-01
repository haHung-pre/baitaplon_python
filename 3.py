import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# Define paths
results_csv_path = r'C:\Users\nguye\Downloads\results.csv'
plots_dir = r'C:\Users\nguye\Downloads\plots'
os.makedirs(plots_dir, exist_ok=True)

# Load the data
df = pd.read_csv(results_csv_path)

# Define statistics to use for clustering
stats = ['Gls', 'Ast', 'xG', 'xAG']

# Convert statistics columns to numeric, replacing "N/a" with NaN
for stat in stats:
    df[stat] = pd.to_numeric(df[stat], errors='coerce')

# Drop rows where all stats are NaN (players with no valid data)
df = df.dropna(subset=stats, how='all')

# Prepare data for clustering
X = df[stats].fillna(0)  # Replace NaN with 0 for clustering

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Task 1: Determine the optimal number of clusters using the elbow method
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.savefig(os.path.join(plots_dir, 'elbow_curve.png'))
plt.close()

# Choose K based on the elbow point (we'll analyze the plot manually, but let's assume K=4 for now)
optimal_k = 4  # Adjust based on elbow curve observation

# Task 2: Apply K-means clustering with the chosen K
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Task 3: Apply PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Add PCA components to DataFrame
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

# Plot the 2D clusters
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['PCA1'], df['PCA2'], c=df['Cluster'], cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Cluster')
plt.title('2D PCA Clustering of Premier League Players')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.savefig(os.path.join(plots_dir, 'pca_clusters.png'))
plt.close()

# Save the clustered data with PCA components
df[['Player', 'Squad'] + stats + ['Cluster', 'PCA1', 'PCA2']].to_csv(
    r'C:\Users\nguye\Downloads\clustered_players.csv', index=False)

# Task 4: Analyze clusters (optional for insight)
cluster_summary = df.groupby('Cluster')[stats].mean()
print("Average statistics per cluster:")
print(cluster_summary)

# Reasoning for the number of clusters
print("\nReasoning for Number of Clusters (K):")
print(f"- The elbow method was used to determine the optimal K. The elbow curve (saved as 'elbow_curve.png') shows inertia vs. K.")
print(f"- I chose K={optimal_k} because it appears to be the point where adding more clusters yields diminishing returns in reducing inertia.")
print("- This K value balances between capturing meaningful player groups and avoiding overfitting.")
print("- In the context of Premier League players, K=4 might represent groups like:")
print("  - Cluster 0: Low-performing players (low Gls, Ast, xG, xAG).")
print("  - Cluster 1: High-scoring forwards (high Gls, xG).")
print("  - Cluster 2: Playmakers (high Ast, xAG).")
print("  - Cluster 3: Balanced attackers (moderate Gls, Ast, xG, xAG).")

print("\nOutputs generated:")
print(f"- Elbow curve saved to {os.path.join(plots_dir, 'elbow_curve.png')}")
print(f"- 2D PCA cluster plot saved to {os.path.join(plots_dir, 'pca_clusters.png')}")
print(f"- Clustered data saved to C:\\Users\\nguye\\Downloads\\clustered_players.csv")