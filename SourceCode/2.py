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

# Load data
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
    for team in valid_teams:
        if team in squad:
            return team
    return squad

df['Squad'] = df['Squad'].apply(clean_squad)

# Define statistics
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

# Convert stats to numeric before normalization
for stat in stats:
    df[stat] = pd.to_numeric(df[stat], errors='coerce')

# Debug: Check for non-numeric columns
non_numeric_stats = [stat for stat in stats if df[stat].dtype not in ['float64', 'int64']]
if non_numeric_stats:
    print(f"Warning: These stats could not be converted to numeric: {non_numeric_stats}")

# Normalize team-level stats by player count (only for numeric, non-rate stats)
player_counts = df.groupby('Squad')['Player'].count()
numeric_stats = [
    stat for stat in stats 
    if stat not in ['Gls_per90', 'Ast_per90', 'xG_per90', 'xAG_per90', 'GA90', 'Save%', 'CS%', 'PK Save%', 
                    'SoT%', 'Total Cmp%', 'Short Cmp%', 'Medium Cmp%', 'Long Cmp%', 'Succ%', 'Tkld%', 'Won%']
    and df[stat].dtype in ['float64', 'int64']
]
for stat in numeric_stats:
    df[stat] = df.apply(lambda x: x[stat] / player_counts[x['Squad']] if pd.notna(x[stat]) else x[stat], axis=1)

# Debug: Print unique Squad values and number of players per team
print("Unique Squad values before filtering:", df['Squad'].unique())
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

# Drop rows where all stats are NaN
df = df.dropna(subset=stats, how='all')

# Function to sanitize filenames
def sanitize_filename(name):
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
    top_low = stat_df.nsmallest(3, stat)
    top_3_content.append(f"Top 3 lowest for {stat}:\n")
    for _, row in top_low.iterrows():
        top_3_content.append(f"{row['Player']}: {row[stat]:.2f}\n")
    top_3_content.append("\n")

# Save to top_3.txt
with open(top_3_path, 'w', encoding='utf-8') as f:
    f.writelines(top_3_content)

# Task 2: Calculate median, mean, and standard deviation for each team
stats_summary = []
for team in valid_teams:
    team_df = df[df['Squad'] == team]
    row = {'Team': team}
    for stat in stats:
        valid_data = team_df[stat].dropna()
        row[f'Median of {stat}'] = valid_data.median() if len(valid_data) > 0 else np.nan
        row[f'Mean of {stat}'] = valid_data.mean() if len(valid_data) > 0 else np.nan
        row[f'Std of {stat}'] = valid_data.std() if len(valid_data) > 0 else np.nan
    stats_summary.append(row)

# Create DataFrame and save to results2.csv
stats_df = pd.DataFrame(stats_summary)
# Do not fill NaN with 'N/a' to keep empty cells in CSV
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

goalkeeper_stats = ['GA90', 'Save%', 'CS%', 'PK Save%']
for team in valid_teams:
    team_df = df[df['Squad'] == team]
    sanitized_team = sanitize_filename(team)
    skipped_stats = [stat for stat in stats if stat not in goalkeeper_stats and len(team_df[stat].dropna()) < 5]
    if skipped_stats:
        print(f"Skipping histograms for {', '.join(skipped_stats)} in {team}: insufficient data")
    for stat in stats:
        if stat in goalkeeper_stats:
            valid_data = team_df[team_df['Pos'] == 'GK'][stat].dropna()
        else:
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
        top_teams[stat] = ('None', np.nan)

# Save top teams to file
with open(top_3_path, 'a', encoding='utf-8') as f:
    f.write("\nTeams with highest average scores for each statistic:\n")
    for stat, (team, score) in top_teams.items():
        score_str = f"{score:.2f}" if isinstance(score, (int, float)) and not np.isnan(score) else 'N/a'
        f.write(f"{stat}: {team} with average {score_str}\n")

# Task 5: Analysis of best-performing team
team_scores = team_means.mean(axis=1, skipna=True)
valid_scores = team_scores.dropna()
if len(valid_scores) > 0:
    best_team = valid_scores.idxmax()
    best_team_score = valid_scores.max()
else:
    best_team = 'None'
    best_team_score = np.nan

score_str = f"{best_team_score:.2f}" if isinstance(best_team_score, (int, float)) and not np.isnan(best_team_score) else 'N/a'
print(f"\nAnalysis of Best-Most-Performing Team:")
print(f"The team with the highest overall average across all statistics is {best_team} with an average score of {score_str}.")
print("Reasoning:")
offensive = [stat for stat, (team, _) in top_teams.items() if team == best_team and stat in ['Gls', 'Ast', 'xG', 'xAG']]
defensive = [stat for stat, (team, _) in top_teams.items() if team == best_team and stat in ['Tkl', 'Int', 'Blocks']]
possession = [stat for stat, (team, _) in top_teams.items() if team == best_team and stat in ['Touches', 'Carries', 'PrgC_possession']]
print(f"- Excels in offensive metrics: {', '.join(offensive) or 'None'}")
print(f"- Strong in defensive metrics: {', '.join(defensive) or 'None'}")
print(f"- Effective in possession metrics: {', '.join(possession) or 'None'}")
print(f"- In the 2024-2025 Premier League season, this balanced performance across attack, defense, and possession aligns with a top-performing team.")

print("\nAll tasks completed:")
print(f"- Top 3 players and top teams saved to {top_3_path}")
print(f"- Statistics summary saved to {results2_csv_path}")
print(f"- Histograms saved in {plots_dir}")
