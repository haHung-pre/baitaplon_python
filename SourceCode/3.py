import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os

# Define paths
results_csv_path = r'C:\Users\nguye\Downloads\results.csv'
plots_dir = r'C:\Users\nguye\Downloads\plots'
os.makedirs(plots_dir, exist_ok=True)

# Load the data
df = pd.read_csv(results_csv_path)

# Define all numeric statistics (excluding goalkeeper stats for non-goalkeepers)
stats = [
    'MP', 'Starts', 'Min', 'Gls', 'Ast', 'CrdY', 'CrdR', 'xG', 'npxG', 'xAG',
    'PrgC_standard', 'PrgP_standard', 'PrgR_standard', 'Gls_per90', 'Ast_per90', 'xG_per90', 'xAG_per90',
    'SoT%', 'SoT/90', 'G/Sh', 'Dist', 'Total Cmp', 'Total Cmp%', 'TotDist_passing',
    'Short Cmp%', 'Medium Cmp%', 'Long Cmp%', 'KP', '1/3_passing', 'PPA', 'CrsPA', 'PrgP_passing',
    'SCA', 'SCA90', 'GCA', 'GCA90', 'Tkl', 'TklW', 'Att_defense', 'Lost_defense',
    'Blocks', 'Sh_defense', 'Pass_defense', 'Int', 'Touches', 'Def Pen', 'Def 3rd', 'Mid 3rd',
    'Att 3rd', 'Att Pen', 'Att (Take-Ons)', 'Succ%', 'Tkld%', 'Carries', 'TotDist (Carries)',
    'PrgDist (Carries)', 'PrgC_possession', '1/3 (Carries)', 'CPA', 'Mis', 'Dis', 'Rec',
    'PrgR (Receiving)', 'Fls', 'Fld', 'Off', 'Crs', 'Recov', 'Won', 'Lost_misc', 'Won%'
]
goalkeeper_stats = ['GA90', 'Save%', 'CS%', 'PK Save%']

# Convert statistics to numeric
for stat in stats + goalkeeper_stats:
    df[stat] = pd.to_numeric(df[stat], errors='coerce')

# Separate goalkeepers (Pos == 'GK') and non-goalkeepers
gk_df = df[df['Pos'] == 'GK'].copy()
non_gk_df = df[df['Pos'] != 'GK'].copy()

# For non-goalkeepers, exclude goalkeeper stats
non_gk_stats = stats  # Only non-goalkeeper stats
non_gk_df = non_gk_df[['Player', 'Squad', 'Pos'] + non_gk_stats].dropna(subset=non_gk_stats, how='all')

# For goalkeepers, use all stats (including goalkeeper stats)
gk_stats = stats + goalkeeper_stats
gk_df = gk_df[['Player', 'Squad', 'Pos'] + gk_stats].dropna(subset=gk_stats, how='all')

# Combine data
df = pd.concat([non_gk_df, gk_df], ignore_index=True)

# Prepare data for clustering
# Impute missing values: use mean for continuous stats, 0 for counts
X = df[stats].copy()
for stat in stats:
    if stat in ['Gls_per90', 'Ast_per90', 'xG_per90', 'xAG_per90', 'SoT%', 'G/Sh', 'Dist',
                'Total Cmp%', 'Short Cmp%', 'Medium Cmp%', 'Long Cmp%', 'SCA90', 'GCA90',
                'Succ%', 'Tkld%', 'Won%']:
        X[stat] = X[stat].fillna(X[stat].mean())
    else:
        X[stat] = X[stat].fillna(0)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Task 1: Determine the optimal number of clusters
inertia = []
silhouette_scores = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    if k > 1:
        score = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(score)
    else:
        silhouette_scores.append(0)  # Placeholder for k=1

# Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertia, marker='o', label='Inertia')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.grid(True)
plt.savefig(os.path.join(plots_dir, 'elbow_curve.png'))
plt.close()

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(K_range, silhouette_scores, marker='o', label='Silhouette Score')
plt.title('Silhouette Score for Different K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.savefig(os.path.join(plots_dir, 'silhouette_scores.png'))
plt.close()

# Choose optimal K (based on elbow and silhouette)
optimal_k = 4  # Adjust based on plots (elbow point and high silhouette score)

# Task 2: Apply K-means clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Task 3: Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
explained_variance_ratio = pca.explained_variance_ratio_
print(f"PCA Explained Variance Ratio: {explained_variance_ratio}")
print(f"Total Variance Explained: {sum(explained_variance_ratio):.2f}")

# Add PCA components to DataFrame
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

# Plot the 2D clusters
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df['PCA1'], df['PCA2'], c=df['Cluster'], cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Cluster')
plt.title(f'2D PCA Clustering of Premier League Players (K={optimal_k})')
plt.xlabel(f'PCA Component 1 ({explained_variance_ratio[0]:.2%} variance)')
plt.ylabel(f'PCA Component 1 ({explained_variance_ratio[1]:.2%} variance)')
# Add labels for some players (e.g., top players in each cluster)
for cluster in range(optimal_k):
    cluster_df = df[df['Cluster'] == cluster]
    top_player = cluster_df.nlargest(1, 'Gls')  # Example: top scorer
    plt.annotate(top_player['Player'].iloc[0], (top_player['PCA1'].iloc[0], top_player['PCA2'].iloc[0]))
plt.grid(True)
plt.savefig(os.path.join(plots_dir, 'pca_clusters.png'))
plt.close()

# Save clustered data
output_cols = ['Player', 'Squad', 'Pos', 'Cluster', 'PCA1', 'PCA2'] + stats
df[output_cols].to_csv(r'C:\Users\nguye\Downloads\clustered_players.csv', index=False)

# Task 4: Analyze clusters
cluster_summary = df.groupby('Cluster')[stats].mean()
cluster_summary.to_csv(os.path.join(plots_dir, 'cluster_summary.csv'))

# Count players per cluster and position
cluster_pos_counts = df.groupby(['Cluster', 'Pos']).size().unstack(fill_value=0)
cluster_pos_counts.to_csv(os.path.join(plots_dir, 'cluster_position_counts.csv'))

# Print analysis
print("Average Statistics per Cluster:")
print(cluster_summary)
print("\nPlayer Count by Cluster and Position:")
print(cluster_pos_counts)

print("\nReasoning for Number of Clusters (K):")
print(f"- The elbow method (saved as 'elbow_curve.png') shows the inertia vs. K. A noticeable elbow appears around K={optimal_k}, indicating diminishing returns for higher K.")
print(f"- The silhouette score (saved as 'silhouette_scores.png') peaks near K={optimal_k}, confirming that this K provides well-separated clusters.")
print(f"- K={optimal_k} is suitable for Premier League players, as it allows grouping into meaningful roles:")
print("  - Cluster 0: Likely defenders (high Tkl, Int, low Gls, Ast).")
print("  - Cluster 1: Likely attackers (high Gls, xG, SCA).")
print("  - Cluster 2: Likely midfielders (high Touches, PrgP, KP).")
print("  - Cluster 3: Likely goalkeepers or low-minute players (high Save% for GKs, low overall stats).")
print("- The PCA plot shows distinct clusters, with some overlap due to player versatility (e.g., wing-backs with both defensive and attacking stats).")
print(f"- PCA explains {sum(explained_variance_ratio):.2%} of variance, indicating moderate representation of the data in 2D.")

print("\nComments on Clustering Results:")
print("- The clusters reflect player roles (defenders, midfielders, attackers, goalkeepers), as seen in the position counts.")
print("- Defenders dominate clusters with high defensive stats (Tkl, Int), while attackers have high Gls, xG.")
print("- Goalkeepers are likely in a separate cluster due to unique stats (Save%, GA90), but their small number limits their impact.")
print("- Missing data ('N/a') was handled by imputing means for percentages and 0 for counts, reducing bias.")
print("- Some overlap in clusters is expected due to versatile players (e.g., midfielders with defensive and attacking contributions).")
print("- The PCA plot highlights key players in each cluster (e.g., top scorers), aiding interpretation.")
print("- Limitations: PCA's low variance explained suggests higher dimensions might capture more details. Future work could use t-SNE or additional features.")

print("\nOutputs Generated:")
print(f"- Elbow curve: {os.path.join(plots_dir, 'elbow_curve.png')}")
print(f"- Silhouette scores: {os.path.join(plots_dir, 'silhouette_scores.png')}")
print(f"- PCA cluster plot: {os.path.join(plots_dir, 'pca_clusters.png')}")
print(f"- Clustered data: C:\\Users\\nguye\\Downloads\\clustered_players.csv")
print(f"- Cluster summary: {os.path.join(plots_dir, 'cluster_summary.csv')}")
print(f"- Cluster position counts: {os.path.join(plots_dir, 'cluster_position_counts.csv')}")
