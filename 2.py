import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Define paths
results_csv_path = r'C:\Users\nguye\Downloads\results.csv'
top_3_path = r'C:\Users\nguye\Downloads\top_3.txt'
results2_csv_path = r'C:\Users\nguye\Downloads\results2.csv'
plots_dir = r'C:\Users\nguye\Downloads\plots'
os.makedirs(plots_dir, exist_ok=True)

# Load the data
df = pd.read_csv(results_csv_path)

# Define statistics to analyze
stats = ['Gls', 'Ast', 'xG', 'xAG']

# Convert statistics columns to numeric, replacing "N/a" with NaN
for stat in stats:
    df[stat] = pd.to_numeric(df[stat], errors='coerce')

# Drop rows where all stats are NaN (players with no valid data)
df = df.dropna(subset=stats, how='all')

# Task 1: Identify top 3 highest and lowest for each statistic
top_3_content = []
for stat in stats:
    # Remove NaN for sorting
    stat_df = df[['Player', stat]].dropna()
    
    # Top 3 highest
    top_high = stat_df.nlargest(3, stat)
    top_3_content.append(f"Top 3 highest for {stat}:\n")
    for _, row in top_high.iterrows():
        top_3_content.append(f"{row['Player']}: {row[stat]}\n")
    
    # Top 3 lowest (excluding zeros for meaningful stats)
    stat_df_nonzero = stat_df[stat_df[stat] > 0]
    if len(stat_df_nonzero) >= 3:
        top_low = stat_df_nonzero.nsmallest(3, stat)
    else:
        top_low = stat_df.nsmallest(3, stat)
    top_3_content.append(f"Top 3 lowest for {stat}:\n")
    for _, row in top_low.iterrows():
        top_3_content.append(f"{row['Player']}: {row[stat]}\n")
    top_3_content.append("\n")

# Save to top_3.txt
with open(top_3_path, 'w', encoding='utf-8') as f:
    f.writelines(top_3_content)

# Task 2: Calculate median, mean, and standard deviation for each statistic
# Across all players
stats_summary = []
for stat in stats:
    row = {'Team': 'all', f'Median of {stat}': df[stat].median(), 
           f'Mean of {stat}': df[stat].mean(), f'Std of {stat}': df[stat].std()}
    stats_summary.append(row)

# Per team
teams = df['Squad'].unique()
for team in teams:
    team_df = df[df['Squad'] == team]
    row = {'Team': team}
    for stat in stats:
        row[f'Median of {stat}'] = team_df[stat].median()
        row[f'Mean of {stat}'] = team_df[stat].mean()
        row[f'Std of {stat}'] = team_df[stat].std()
    stats_summary.append(row)

# Create DataFrame and save to results2.csv
stats_df = pd.DataFrame(stats_summary)
stats_df = stats_df.fillna('N/a')
stats_df.to_csv(results2_csv_path, index=False)

# Task 3: Plot histograms for each statistic
# For all players
for stat in stats:
    plt.figure(figsize=(10, 6))
    plt.hist(df[stat].dropna(), bins=30, edgecolor='black')
    plt.title(f'Distribution of {stat} for All Players')
    plt.xlabel(stat)
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(plots_dir, f'{stat}_all_players.png'))
    plt.close()

# Per team
for team in teams:
    team_df = df[df['Squad'] == team]
    for stat in stats:
        plt.figure(figsize=(10, 6))
        plt.hist(team_df[stat].dropna(), bins=30, edgecolor='black')
        plt.title(f'Distribution of {stat} for {team}')
        plt.xlabel(stat)
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(plots_dir, f'{stat}_{team.replace(" ", "_")}.png'))
        plt.close()

# Task 4: Identify team with highest scores for each statistic
team_means = df.groupby('Squad')[stats].mean()
top_teams = {}
for stat in stats:
    top_team = team_means[stat].idxmax()
    top_score = team_means[stat].max()
    top_teams[stat] = (top_team, top_score)

# Print analysis
print("Teams with highest average scores for each statistic:")
for stat, (team, score) in top_teams.items():
    print(f"{stat}: {team} with average {score:.2f}")

# Task 5: Analysis of best-performing team
# Based on the means, identify the team that consistently performs well
team_scores = team_means.mean(axis=1)  # Average of means across all stats
best_team = team_scores.idxmax()
best_team_score = team_scores.max()

print(f"\nAnalysis of Best-Performing Team:")
print(f"The team with the highest overall average across all statistics is {best_team} with an average score of {best_team_score:.2f}.")
print("Reasoning:")
print(f"- {best_team} shows strong performance across key offensive metrics (Gls, Ast, xG, xAG), indicating a balanced and effective attacking strategy.")
print(f"- High xG and xAG suggest they create high-quality chances, while Gls and Ast show they convert these chances effectively.")
print(f"- In the 2024-2025 Premier League season, this aligns with teams that have top players and tactical consistency.")

print("\nAll tasks completed:")
print(f"- Top 3 players saved to {top_3_path}")
print(f"- Statistics summary saved to {results2_csv_path}")
print(f"- Histograms saved in {plots_dir}")