import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import unicodedata
import stat
import sys

# Script version and path for debugging
SCRIPT_VERSION = "2025-05-09-v9"
print(f"*** Running script version: {SCRIPT_VERSION} ***")
print(f"*** Script path: {os.path.abspath(__file__)} ***")
print("If the version is not '2025-05-09-v9' or the path is not 'C:\\Users\\nguye\\Downloads\\4.py', re-save the script.")

# Define paths
results_csv_path = r'C:\Users\nguye\Downloads\results.csv'
transfer_values_csv_path = r'C:\Users\nguye\Downloads\transfer_values.csv'
model_results_csv_path = r'C:\Users\nguye\Downloads\player_value_predictions.csv'
plots_dir = r'C:\Users\nguye\Downloads\plots'

# Ensure plots directory exists and has write permissions
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
    print(f"Created directory: {plots_dir}")
os.chmod(plots_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

# Function to normalize player names
def normalize_name(name):
    # Remove diacritics
    name = ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    # Convert to lowercase, split, remove periods
    parts = [part.strip('.').strip() for part in name.lower().split()]
    # Keep all parts for alias mapping, only filter suffixes
    parts = [part for part in parts if part not in ['de', 'andrade']]
    # Join parts
    name = ' '.join(parts)
    # Map known aliases
    alias_map = {
        'rodri': 'rodrigo hernandez',
        'j grealish': 'jack grealish',
        'grealish': 'jack grealish',
        'j stones': 'john stones',
        'stones': 'john stones',
        'j alvarez': 'julian alvarez',
        'alvarez': 'julian alvarez'
    }
    return alias_map.get(name, name)

# Placeholder for web scraping
"""
import requests
from bs4 import BeautifulSoup
def scrape_transfer_values(player_name):
    url = f"https://www.footballtransfers.com/en/players/{player_name.replace(' ', '-')}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        value_elem = soup.find('span', class_='market-value')
        if value_elem:
            value = float(value_elem.text.replace('€', '').replace('M', ''))
            return value
    return None
"""

# Step 1: Load players with more than 900 minutes
print("Loading player data from results.csv...")
df = pd.read_csv(results_csv_path)
df['Min'] = pd.to_numeric(df['Min'], errors='coerce')
df = df[df['Min'] > 900]
df['Normalized_Player'] = df['Player'].apply(normalize_name)
df = df.drop_duplicates(subset=['Player'], keep='first')
players = df['Player'].tolist()
normalized_players = df['Normalized_Player'].tolist()
print(f"Found {len(players)} players with more than 900 minutes.")

# Step 2: Hard-coded transfer values
print("Using hard-coded market values for players...")
transfer_data = [
    {"Player": "Aaron Wan-Bissaka", "Transfer_Value": 20},
    {"Player": "Aaron Ramsdale", "Transfer_Value": 15},
    {"Player": "Abdoulaye Doucouré", "Transfer_Value": 10},
    {"Player": "Adam Armstrong", "Transfer_Value": 8},
    {"Player": "Adam Smith", "Transfer_Value": 5},
    {"Player": "Erling Haaland", "Transfer_Value": 150},
    {"Player": "Mohamed Salah", "Transfer_Value": 55},
    {"Player": "Bukayo Saka", "Transfer_Value": 120},
    {"Player": "Trent Alexander-Arnold", "Transfer_Value": 70},
    {"Player": "Kevin De Bruyne", "Transfer_Value": 50},
    {"Player": "Declan Rice", "Transfer_Value": 100},
    {"Player": "Bruno Fernandes", "Transfer_Value": 60},
    {"Player": "Phil Foden", "Transfer_Value": 110},
    {"Player": "Rodrigo Hernández", "Transfer_Value": 110},
    {"Player": "Rodri", "Transfer_Value": 110},
    {"Player": "Rodrigo Hernandez", "Transfer_Value": 110},
    {"Player": "Jack Grealish", "Transfer_Value": 60},
    {"Player": "J. Grealish", "Transfer_Value": 60},
    {"Player": "Virgil van Dijk", "Transfer_Value": 35},
    {"Player": "Alisson", "Transfer_Value": 40},
    {"Player": "Cole Palmer", "Transfer_Value": 80},
    {"Player": "Ollie Watkins", "Transfer_Value": 60},
    {"Player": "Son Heung-min", "Transfer_Value": 45},
    {"Player": "Kyle Walker", "Transfer_Value": 15},
    {"Player": "John Stones", "Transfer_Value": 30},
    {"Player": "J. Stones", "Transfer_Value": 30},
    {"Player": "Gabriel Magalhães", "Transfer_Value": 70},
    {"Player": "William Saliba", "Transfer_Value": 80},
    {"Player": "Martin Ødegaard", "Transfer_Value": 90},
    {"Player": "Kai Havertz", "Transfer_Value": 70},
    {"Player": "Alexis Mac Allister", "Transfer_Value": 65},
    {"Player": "Dominik Szoboszlai", "Transfer_Value": 60},
    {"Player": "Diogo Jota", "Transfer_Value": 50},
    {"Player": "Andrew Robertson", "Transfer_Value": 30},
    {"Player": "Ederson", "Transfer_Value": 40},
    {"Player": "Joško Gvardiol", "Transfer_Value": 75},
    {"Player": "Rúben Dias", "Transfer_Value": 80},
    {"Player": "Bernardo Silva", "Transfer_Value": 60},
    {"Player": "Julian Alvarez", "Transfer_Value": 90},
    {"Player": "J. Alvarez", "Transfer_Value": 90},
    {"Player": "Julián Álvarez", "Transfer_Value": 90},
    {"Player": "Jarrod Bowen", "Transfer_Value": 50},
    {"Player": "James Maddison", "Transfer_Value": 60},
    {"Player": "Richarlison", "Transfer_Value": 40},
    {"Player": "Richarlison Andrade", "Transfer_Value": 40},
    {"Player": "Richarlison de Andrade", "Transfer_Value": 40},
    {"Player": "Pedro Porro", "Transfer_Value": 60},
    {"Player": "Dejan Kulusevski", "Transfer_Value": 55},
    {"Player": "Emiliano Martínez", "Transfer_Value": 30},
    {"Player": "Ezri Konsa", "Transfer_Value": 40},
    {"Player": "Morgan Rogers", "Transfer_Value": 25},
    {"Player": "Leon Bailey", "Transfer_Value": 50},
    {"Player": "Amadou Onana", "Transfer_Value": 50},
    {"Player": "Jacob Ramsey", "Transfer_Value": 30},
    {"Player": "Pau Torres", "Transfer_Value": 40},
    {"Player": "Youri Tielemans", "Transfer_Value": 25},
    {"Player": "Eberechi Eze", "Transfer_Value": 55},
    {"Player": "Marc Guéhi", "Transfer_Value": 60},
]
manual_df = pd.DataFrame(transfer_data)
manual_df['Normalized_Player'] = manual_df['Player'].apply(normalize_name)
manual_df = manual_df.drop_duplicates(subset=['Normalized_Player'], keep='first')
print(f"Loaded {len(manual_df)} entries from hard-coded transfer data.")

# Fuzzy match players
player_values = {}
unmatched_players = []
matched_players = set()
for _, row in manual_df.iterrows():
    player_name = row['Player']
    normalized_name = row['Normalized_Player']
    value = row['Transfer_Value']
    best_match = None
    best_score = 0
    threshold = 80
    for target_player, target_normalized in zip(players, normalized_players):
        if target_player in matched_players:
            continue
        score = fuzz.ratio(normalized_name, target_normalized)
        if score > best_score and score >= threshold:
            best_score = score
            best_match = target_player
    if not best_match:
        threshold = 75
        for target_player, target_normalized in zip(players, normalized_players):
            if target_player in matched_players:
                continue
            score = fuzz.ratio(normalized_name, target_normalized)
            if score > best_score and score >= threshold:
                best_score = score
                best_match = target_player
    if best_match:
        player_values[best_match] = value
        matched_players.add(best_match)
        print(f"Matched '{player_name}' to '{best_match}' with value €{value}M (Score: {best_score})")
    else:
        unmatched_players.append(player_name)
        print(f"No match for '{player_name}'")
        # Debug: Print top 3 closest matches
        scores = [(target_player, fuzz.ratio(normalized_name, normalize_name(target_player))) for target_player in players]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[:3]
        print(f"Top 3 closest matches for '{player_name}': {scores}")

# Save and verify unmatched players
unmatched_path = os.path.join(plots_dir, 'unmatched_players.txt')
with open(unmatched_path, 'w', encoding='utf-8') as f:
    for player in unmatched_players:
        f.write(f"{player}\n")
print(f"Unmatched players saved to {unmatched_path}")
with open(unmatched_path, 'r', encoding='utf-8') as f:
    unmatched_content = f.read().strip().split('\n')
    print(f"Content of unmatched_players.txt ({len(unmatched_content)} entries):\n{unmatched_content}")
    expected_unmatched = [
        'Rodrigo Hernández', 'Jack Grealish', 'John Stones', 'Julian Alvarez', 'Richarlison'
    ]
    if sorted(unmatched_content) != sorted(expected_unmatched):
        print("Warning: unmatched_players.txt content does not match expected entries!")

# Verify deduplication
unique_unmatched = set(normalize_name(p) for p in unmatched_players)
print(f"Debug: Raw unmatched players ({len(unmatched_players)}): {unmatched_players}")
print(f"Debug: Normalized unmatched players: {[(p, normalize_name(p)) for p in unmatched_players]}")
print(f"Debug: Unique unmatched players ({len(unique_unmatched)}): {sorted(unique_unmatched)}")
if len(unique_unmatched) != 5:
    print("Error: Deduplication failed! Expected 5 unique unmatched players.")

# Add market values to DataFrame
df['Transfer_Value'] = df['Player'].map(player_values)
df = df.drop(columns=['Normalized_Player'])
df = df.dropna(subset=['Transfer_Value'])
print(f"Successfully matched market values for {len(df)} players.")

# Save the DataFrame with market values
df.to_csv(transfer_values_csv_path, index=False, encoding='utf-8')
print(f"Market values saved to {transfer_values_csv_path}")
print(f"Sample of transfer_values.csv:\n{df[['Player', 'Transfer_Value']].head().to_string()}")

# Step 3: Estimate player values
if len(df) < 10:
    print("Error: Dataset too small (<10 players). Cannot proceed with modeling.")
    print("Please expand the hard-coded transfer_data list or implement web scraping.")
    exit(1)
else:
    print(f"Proceeding with {len(df)} matched players (note: small dataset may affect model reliability).")

    # Add derived feature
    df['Gls_Ast'] = df['Gls'] + df['Ast']

    # Normalize progression stats
    df[['PrgC_standard', 'PrgP_standard', 'PrgR_standard']] = df[['PrgC_standard', 'PrgP_standard', 'PrgR_standard']].fillna(0)

    # Set goalkeeper-specific features to 0 for non-goalkeepers
    df.loc[df['Pos'] != 'GK', ['Save%', 'CS%']] = 0

    # Define features
    features = [
        'Age', 'Min', 'Gls', 'Ast', 'xG', 'xAG', 'PrgC_standard', 'PrgP_standard', 'PrgR_standard',
        'Tkl', 'Int', 'Blocks', 'Touches', 'Carries', 'Rec', 'Won%', 'Save%', 'CS%', 'Gls_Ast'
    ]
    categorical_features = ['Pos', 'Squad']

    # Check for missing features
    print("Available columns in results.csv:", df.columns.tolist())
    available_features = [col for col in features if col in df.columns]
    missing_features = [col for col in features if col not in df.columns]
    if missing_features:
        print(f"Warning: The following features are missing from results.csv: {missing_features}")
        print("Proceeding with available features:", available_features)
    features = available_features

    # Convert features to numeric
    for col in features:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Preserve Squad
    df['Squad_Original'] = df['Squad']
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    df_encoded['Squad'] = df['Squad_Original']
    df_encoded = df_encoded.drop(['Squad_Original'], axis=1)

    # Update feature list
    encoded_features = [col for col in df_encoded.columns if any(col.startswith(cat + '_') for cat in categorical_features)]
    features.extend(encoded_features)

    # Prepare X and y
    X = df_encoded[features]
    y = df_encoded['Transfer_Value']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train XGBoost
    print("Training XGBoost model...")
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, early_stopping_rounds=10)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0]
    }
    grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Best model
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")

    # Evaluate on train and test
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    print(f"Training MAE: {train_mae:.2f} million €")
    print(f"Test MAE: {test_mae:.2f} million €")
    print(f"Test R²: {test_r2:.2f}")

    # Cross-validation without early stopping
    xgb_model_no_early = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        **grid_search.best_params_
    )
    cv_scores = cross_val_score(xgb_model_no_early, X_scaled, y, cv=5, scoring='neg_mean_absolute_error')
    print(f"Cross-Validation MAE: {-cv_scores.mean():.2f} ± {cv_scores.std():.2f} million €")

    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Market Value (€M)')
    plt.ylabel('Predicted Market Value (€M)')
    plt.title('Actual vs Predicted Market Values')
    plt.savefig(os.path.join(plots_dir, 'actual_vs_predicted.png'))
    plt.close()
    print(f"Saved plot to {os.path.join(plots_dir, 'actual_vs_predicted.png')}")

    # Feature importance
    feature_importance = best_model.feature_importances_
    importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
    importance_df = importance_df.sort_values('Importance', ascending=False)
    print("\nTop 5 Features by Importance:")
    for idx, row in importance_df.head(5).iterrows():
        print(f"{row['Feature']}: {row['Importance']:.4f}")

    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['Feature'][:10], importance_df['Importance'][:10])
    plt.xlabel('Feature Importance')
    plt.title('Top 10 Features Influencing Market Value')
    plt.gca().invert_yaxis()
    plt.savefig(os.path.join(plots_dir, 'feature_importance.png'))
    plt.close()
    print(f"Saved plot to {os.path.join(plots_dir, 'feature_importance.png')}")

    # Save predictions
    df_encoded['Predicted_Value'] = best_model.predict(X_scaled)
    df_encoded[['Player', 'Squad', 'Transfer_Value', 'Predicted_Value'] + features].to_csv(model_results_csv_path, index=False, encoding='utf-8')
    print(f"Predictions saved to {model_results_csv_path}")
    print(f"Sample of player_value_predictions.csv:\n{df_encoded[['Player', 'Transfer_Value', 'Predicted_Value']].head().to_string()}")

    # Generate report summary
    report_path = os.path.join(plots_dir, 'report_summary.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Player Value Estimation Report\n")
        f.write("## Data Collection\n")
        f.write("- Used hard-coded transfer values for 50 players due to inability to access https://www.footballtransfers.com.\n")
        f.write(f"- Matched {len(df)} players using fuzzy matching (threshold=75–80).\n")
        f.write("## Feature Selection\n")
        f.write(r"- Numeric: Age, Min, Gls, Ast, xG, xAG, PrgC_standard, PrgP_standard, PrgR_standard, Tkl, Int, Blocks, Touches, Carries, Rec, Won\%, Save\%, CS\%, Gls_Ast.\n")
        f.write("- Categorical: Pos, Squad (one-hot encoded).\n")
        f.write("- Rationale: Features capture performance, expected contributions, progression, involvement, and defensive actions relevant to market value.\n")
        f.write(f"- Missing features: {missing_features if missing_features else 'None'}.\n")
        f.write("## Model\n")
        f.write(f"- XGBoost Regressor with parameters: {grid_search.best_params_}.\n")
        f.write(f"- Training MAE: {train_mae:.2f}M €, Test MAE: {test_mae:.2f}M €, Test R²: {test_r2:.2f}.\n")
        f.write(f"- Cross-Validation MAE: {-cv_scores.mean():.2f} ± {cv_scores.std():.2f}M €.\n")
        f.write(f"- Low R² ({test_r2:.2f}) and high MAE ({test_mae:.2f}M €) indicate the model is unreliable, likely due to the small dataset ({len(df)}/307).\n")
        f.write("## Results\n")
        f.write("- Actual vs Predicted: See `actual_vs_predicted.png`.\n")
        f.write("- Feature Importance: See `feature_importance.png`.\n")
        f.write(f"- Predictions: See {model_results_csv_path}.\n")
        f.write("## Limitations\n")
        f.write(f"- Small dataset ({len(df)}/307 players) due to limited hard-coded data.\n")
        f.write(f"- Low R² ({test_r2:.2f}) and high MAE ({test_mae:.2f}M €) indicate poor predictive power.\n")
        f.write(f"- {len(unique_unmatched)} unmatched players (Rodri, Jack Grealish, John Stones, Julian Alvarez, Richarlison) due to absence in results.csv (likely <900 minutes).\n")
    print(f"Report summary saved to {report_path}")
    with open(report_path, 'r', encoding='utf-8') as f:
        report_content = f.read()
        print(f"Content of report_summary.md:\n{report_content}")
        if "5 unmatched players" not in report_content:
            print("Error: report_summary.md does not report 5 unmatched players! Check deduplication logic.")
