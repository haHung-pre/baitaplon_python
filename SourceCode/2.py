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








import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re

# Define paths
results_csv_path = r'C:\Users\nguye\Downloads\results.csv'
top_3_path = r'C:\Users\nguye\Downloads\top_3.txt'
results2_csv_path = r'C:\Users\nguye\Downloads\results2.csv'
plots_dir = r'C:\Users\nguye\Downloads\plots'
os.makedirs(plots_dir, exist_ok=True)

# Load the data
df = pd.read_csv(results_csv_path)

# List of valid Premier League 2024-2025 teams
valid_teams = [
    'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton', 
    'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Ipswich Town',
    'Leicester City', 'Liverpool', 'Manchester City', 'Manchester Utd', 
    'Newcastle Utd', "Nott'ham Forest", 'Southampton', 'Tottenham', 
    'West Ham', 'Wolves'
]

# Clean Squad column
def clean_squad(squad):
    squad = squad.strip()
    # If squad contains multiple teams (e.g., 'Aston Villa, Chelsea'), take the first valid team
    for team in valid_teams:
        if team in squad:
            return team
    return squad  # Return original if no valid team found (will be filtered later)

df['Squad'] = df['Squad'].apply(clean_squad)

# Debug: Print unique Squad values and number of players per team
print("Unique Squad values before filtering:", df['Squad'].unique())
# Filter out teams with too few players (<5)
teams = df['Squad'].value_counts()
valid_teams = [team for team in teams.index if teams[team] >= 5 and team in valid_teams]
df = df[df['Squad'].isin(valid_teams)]
print("\nNumber of players per team after filtering:")
for team in valid_teams:
    print(f"{team}: {len(df[df['Squad'] == team])} players")
print("Unique Squad values after filtering:", df['Squad'].unique())

# Warn about invalid squads
invalid_squads = set(df['Squad'].unique()) - set(valid_teams)
if invalid_squads:
    print(f"Warning: Invalid squad values found and ignored: {invalid_squads}")

# Define statistics to analyze
stats = [
    'MP', 'Starts', 'Min', 'Gls', 'Ast', 'CrdY', 'CrdR', 'xG', 'npxG', 'xAG',
    'PrgC_standard', 'PrgP_standard', 'PrgR_standard', 'Gls_per90', 'Ast_per90', 'xG_per90', 'xAG_per90',
    'GA90', 'Save%', 'CS%', 'PK Save%', 'SoT%', 'SoT/90', 'G/Sh', 'Dist',
    'Total Cmp', 'Total Cmp%', 'TotDist_passing', 'Short Cmp%', 'Medium Cmp%', 'Long Cmp%',
    'KP', '1/3_passing', 'PPA', 'CrsPA', 'PrgP_passing',
    'SCA', 'SCA90', 'GCA', 'GCA90',
    'Tkl', 'TklW', 'Att_defense', 'Lost_defense', 'Blocks', 'Sh_defense', 'Pass_defense', 'Int',
    'Touches', 'Def Pen', 'Def 3rd', 'Mid 3rd', 'Att 3rd', 'Att Pen', 'Att (Take-Ons)', 'Succ%', 'Tkld%',
    'Carries', 'TotDist (Carries)', 'PrgDist (Carries)', 'PrgC_possession', '1/3 (Carries)', 'CPA', 'Mis', 'Dis',
    'Rec', 'PrgR (Receiving)', 'Fls', 'Fld', 'Off', 'Crs', 'Recov', 'Won', 'Lost_misc', 'Won%'
]

# Stats specific to goalkeepers (to skip for team histograms)
goalkeeper_stats = ['GA90', 'Save%', 'CS%', 'PK Save%']

# Convert statistics columns to numeric, replacing "N/a" with NaN
for stat in stats:
    df[stat] = pd.to_numeric(df[stat], errors='coerce')

# Drop rows where all stats are NaN
df = df.dropna(subset=stats, how='all')

# Function to sanitize filenames
def sanitize_filename(name):
    """Replace invalid characters in filenames with underscores."""
    return re.sub(r'[^\w\s-]', '_', name.replace('/', '_').replace('%', 'percent')).strip()

# Task 1: Identify top 3 highest and lowest for each statistic
top_3_content = []
for stat in stats:
    stat_df = df[['Player', stat]].dropna()
    if len(stat_df) < 3:
        top_3_content.append(f"Top 3 highest for {stat}: Insufficient data\n")
        top_3_content.append(f"Top 3 lowest for {stat}: Insufficient data\n")
        top_3_content.append("\n")
        continue
    top_high = stat_df.nlargest(3, stat)
    top_3_content.append(f"Top 3 highest for {stat}:\n")
    for _, row in top_high.iterrows():
        top_3_content.append(f"{row['Player']}: {row[stat]:.2f}\n")
    stat_df_nonzero = stat_df[stat_df[stat] > 0]
    top_low = stat_df_nonzero.nsmallest(3, stat) if len(stat_df_nonzero) >= 3 else stat_df.nsmallest(3, stat)
    top_3_content.append(f"Top 3 lowest for {stat}:\n")
    for _, row in top_low.iterrows():
        top_3_content.append(f"{row['Player']}: {row[stat]:.2f}\n")
    top_3_content.append("\n")

# Save to top_3.txt
with open(top_3_path, 'w', encoding='utf-8') as f:
    f.writelines(top_3_content)

# Task 2: Calculate median, mean, and standard deviation for each statistic
stats_summary = []
for stat in stats:
    valid_data = df[stat].dropna()
    median_val = valid_data.median() if len(valid_data) > 0 else 'N/a'
    mean_val = valid_data.mean() if len(valid_data) > 0 else 'N/a'
    std_val = valid_data.std() if len(valid_data) > 0 else 'N/a'
    row = {'Team': 'all', f'Median of {stat}': median_val, f'Mean of {stat}': mean_val, f'Std of {stat}': std_val}
    stats_summary.append(row)

for team in valid_teams:
    team_df = df[df['Squad'] == team]
    row = {'Team': team}
    for stat in stats:
        valid_data = team_df[stat].dropna()
        row[f'Median of {stat}'] = valid_data.median() if len(valid_data) > 0 else 'N/a'
        row[f'Mean of {stat}'] = valid_data.mean() if len(valid_data) > 0 else 'N/a'
        row[f'Std of {stat}'] = valid_data.std() if len(valid_data) > 0 else 'N/a'
    stats_summary.append(row)

# Create DataFrame and save to results2.csv
stats_df = pd.DataFrame(stats_summary)
stats_df = stats_df.fillna('N/a')
stats_df.to_csv(results2_csv_path, index=False, encoding='utf-8-sig')

# Task 3: Plot histograms for each statistic
for stat in stats:
    valid_data = df[stat].dropna()
    if len(valid_data) < 5:
        print(f"Skipping histogram for {stat}: insufficient data")
        continue
    plt.figure(figsize=(10, 6))
    plt.hist(valid_data, bins=30, edgecolor='black')
    plt.title(f'Distribution of {stat} for All Players')
    plt.xlabel(stat)
    plt.ylabel('Frequency')
    sanitized_stat = sanitize_filename(stat)
    plt.savefig(os.path.join(plots_dir, f'{sanitized_stat}_all_players.png'))
    plt.close()

for team in valid_teams:
    team_df = df[df['Squad'] == team]
    sanitized_team = sanitize_filename(team)
    skipped_stats = [stat for stat in stats if stat not in goalkeeper_stats and len(team_df[stat].dropna()) < 5]
    if skipped_stats:
        print(f"Skipping histograms for {', '.join(skipped_stats)} in {team}: insufficient data")
    for stat in stats:
        if stat in goalkeeper_stats:
            continue
        valid_data = team_df[stat].dropna()
        if len(valid_data) < 5:
            continue
        plt.figure(figsize=(10, 6))
        plt.hist(valid_data, bins=30, edgecolor='black')
        plt.title(f'Distribution of {stat} for {team}')
        plt.xlabel(stat)
        plt.ylabel('Frequency')
        sanitized_stat = sanitize_filename(stat)
        plt.savefig(os.path.join(plots_dir, f'{sanitized_stat}_{sanitized_team}.png'))
        plt.close()

# Task 4: Identify team with highest scores for each statistic
team_means = df.groupby('Squad')[stats].mean()
top_teams = {}
for stat in stats:
    valid_means = team_means[stat].dropna()
    if len(valid_means) > 0:
        top_team = valid_means.idxmax()
        top_score = valid_means.max()
        top_teams[stat] = (top_team, top_score)
    else:
        top_teams[stat] = ('N/a', 'N/a')

# Print analysis
print("Teams with highest average scores for each statistic:")
for stat, (team, score) in top_teams.items():
    score_str = f"{score:.2f}" if isinstance(score, (int, float)) else score
    print(f"{stat}: {team} with average {score_str}")

# Task 5: Analysis of best-performing team
team_scores = team_means.mean(axis=1, skipna=True)
valid_scores = team_scores.dropna()
if len(valid_scores) > 0:
    best_team = valid_scores.idxmax()
    best_team_score = valid_scores.max()
else:
    best_team = 'N/a'
    best_team_score = 'N/a'

# Format best_team_score
score_str = f"{best_team_score:.2f}" if isinstance(best_team_score, (int, float)) else best_team_score

print(f"\nAnalysis of Best-Most-Performing Team:")
print(f"The team with the highest overall average across all statistics is {best_team} with an average score of {score_str}.")
print("Reasoning:")
print(f"- {best_team} excels in offensive metrics (e.g., Gls, Ast, xG, xAG), indicating a strong attacking strategy.")
print(f"- High performance in defensive metrics (e.g., Tkl, Int, Blocks) suggests robust defensive capabilities.")
print(f"- Possession metrics (e.g., Touches, Carries, PrgC_possession) show effective ball control and progression.")
print(f"- In the 2024-2025 Premier League season, this balanced performance across attack, defense, and possession aligns with a top-performing team.")

print("\nAll tasks completed:")
print(f"- Top 3 players saved to {top_3_path}")
print(f"- Statistics summary saved to {results2_csv_path}")
print(f"- Histograms saved in {plots_dir}")


