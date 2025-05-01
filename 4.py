import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import unicodedata

# Define paths
results_csv_path = r'C:\Users\nguye\Downloads\results.csv'
transfer_values_csv_path = r'C:\Users\nguye\Downloads\transfer_values.csv'
model_results_csv_path = r'C:\Users\nguye\Downloads\player_value_predictions.csv'
plots_dir = r'C:\Users\nguye\Downloads\plots'
os.makedirs(plots_dir, exist_ok=True)

# Function to normalize player names
def normalize_name(name):
    """
    Normalize a player name by removing accents, converting to lowercase, and standardizing spaces.
    
    Args:
        name (str): The player name to normalize.
    
    Returns:
        str: The normalized name.
    """
    # Remove accents
    name = ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    # Convert to lowercase and remove extra spaces
    name = ' '.join(name.lower().split())
    return name

# Step 1: Load players with more than 900 minutes
print("Loading player data from results.csv...")
df = pd.read_csv(results_csv_path)
df['Min'] = pd.to_numeric(df['Min'], errors='coerce')
df = df[df['Min'] > 900]  # Filter for players with more than 900 minutes
df['Normalized_Player'] = df['Player'].apply(normalize_name)
# Remove duplicates based on 'Player' to avoid repeated comparisons
df = df.drop_duplicates(subset=['Player'], keep='first')
players = df['Player'].tolist()
normalized_players = df['Normalized_Player'].tolist()
print(f"Found {len(players)} players with more than 900 minutes.")
print("Sample player names from results.csv:", players[:5])

# Step 2: Use hard-coded transfer values instead of loading from a CSV
print("Using hard-coded market values for players...")
# Hard-coded dataset of transfer values (50 players)
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
    {"Player": "Rodri", "Transfer_Value": 110},
    {"Player": "Jack Grealish", "Transfer_Value": 60},
    {"Player": "Virgil van Dijk", "Transfer_Value": 35},
    {"Player": "Alisson Becker", "Transfer_Value": 40},
    {"Player": "Cole Palmer", "Transfer_Value": 80},
    {"Player": "Ollie Watkins", "Transfer_Value": 60},
    {"Player": "Son Heung-min", "Transfer_Value": 45},
    {"Player": "Kyle Walker", "Transfer_Value": 15},
    {"Player": "John Stones", "Transfer_Value": 30},
    {"Player": "Gabriel Magalhães", "Transfer_Value": 70},
    {"Player": "William Saliba", "Transfer_Value": 80},
    {"Player": "Martin Ødegaard", "Transfer_Value": 90},
    {"Player": "Kai Havertz", "Transfer_Value": 70},
    {"Player": "Alexis Mac Allister", "Transfer_Value": 65},
    {"Player": "Dominik Szoboszlai", "Transfer_Value": 60},
    {"Player": "Diogo Jota", "Transfer_Value": 50},
    {"Player": "Andrew Robertson", "Transfer_Value": 30},
    {"Player": "Ederson Moraes", "Transfer_Value": 40},
    {"Player": "Josko Gvardiol", "Transfer_Value": 75},
    {"Player": "Ruben Dias", "Transfer_Value": 80},
    {"Player": "Bernardo Silva", "Transfer_Value": 60},
    {"Player": "Julian Alvarez", "Transfer_Value": 90},
    {"Player": "Jarrod Bowen", "Transfer_Value": 50},
    {"Player": "James Maddison", "Transfer_Value": 60},
    {"Player": "Richarlison", "Transfer_Value": 40},
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

# Convert hard-coded data to a DataFrame
manual_df = pd.DataFrame(transfer_data)

# Normalize names in manual data
manual_df['Normalized_Player'] = manual_df['Player'].apply(normalize_name)
print(f"Loaded {len(manual_df)} entries from hard-coded transfer data.")
print("Sample player names from transfer data:", manual_df['Player'].tolist()[:5])

# Fuzzy match players from manual data with players from results.csv
player_values = {}
for _, row in manual_df.iterrows():
    player_name = row['Player']
    normalized_name = row['Normalized_Player']
    value = row['Transfer_Value']
    
    best_match = None
    best_score = 0
    for target_player, target_normalized in zip(players, normalized_players):
        score = fuzz.ratio(normalized_name, target_normalized)
        print(f"Comparing '{player_name}' to '{target_player}': Score = {score}")
        if score > best_score and score > 60:  # Threshold for a good match
            best_score = score
            best_match = target_player
    
    if best_match:
        player_values[best_match] = value
        print(f"Matched '{player_name}' to '{best_match}' with market value €{value}M (Score: {best_score})")
    else:
        print(f"No match found for '{player_name}'")

# Add market values to DataFrame
df['Transfer_Value'] = df['Player'].map(player_values)
df = df.drop(columns=['Normalized_Player'])  # Clean up temporary column
df = df.dropna(subset=['Transfer_Value'])  # Remove players without transfer values
print(f"Successfully matched market values for {len(df)} players.")

# Save the DataFrame with market values
df.to_csv(transfer_values_csv_path, index=False)
print(f"Market values saved to {transfer_values_csv_path}")

# Step 3: Propose and implement a method to estimate player values
if len(df) == 0:
    print("Error: No players with market values were found. Cannot proceed with modeling.")
    print("Please ensure the hard-coded transfer data contains matching player names.")
    print("You may need to adjust player names in the transfer_data list to match those in results.csv.")
    print("Alternatively, lower the fuzzy matching threshold further in the script.")
    exit(1)

# Define features for the model
features = [
    'Age', 'Min', 'Gls', 'Ast', 'xG', 'xAG', 'PrgC', 'PrgP', 'PrgR',
    'SCA', 'GCA', 'Tkl', 'Int', 'Blocks', 'Touches', 'Carries', 'Rec'
]
categorical_features = ['Pos', 'Squad']

# Convert numerical features to numeric, replacing "N/a" with 0
for col in features:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Preserve the Squad column before encoding
df['Squad_Original'] = df['Squad']

# Encode categorical features
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# Restore the Squad column
df_encoded['Squad'] = df['Squad_Original']
df_encoded = df_encoded.drop(columns=['Squad_Original'])

# Update feature list to include encoded categorical features
encoded_features = [col for col in df_encoded.columns if any(col.startswith(cat + '_') for cat in categorical_features)]
features.extend(encoded_features)

# Prepare X and y
X = df_encoded[features]
y = df_encoded['Transfer_Value']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train XGBoost model with hyperparameter tuning
print("Training XGBoost model...")
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1]
}
grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Predict on test set
y_pred = best_model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Model Performance on Test Set:")
print(f"Mean Absolute Error: {mae:.2f} million €")
print(f"R² Score: {r2:.2f}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Market Value (€M)')
plt.ylabel('Predicted Market Value (€M)')
plt.title('Actual vs Predicted Market Values')
plt.savefig(os.path.join(plots_dir, 'actual_vs_predicted.png'))
plt.close()

# Feature importance
feature_importance = best_model.feature_importances_
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
importance_df = importance_df.sort_values('Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 8))
plt.barh(importance_df['Feature'][:10], importance_df['Importance'][:10])
plt.xlabel('Feature Importance')
plt.title('Top 10 Features Influencing Market Value')
plt.gca().invert_yaxis()
plt.savefig(os.path.join(plots_dir, 'feature_importance.png'))
plt.close()

# Add predictions to DataFrame and save
df_encoded['Predicted_Value'] = best_model.predict(X_scaled)
df_encoded[['Player', 'Squad', 'Transfer_Value', 'Predicted_Value'] + features].to_csv(model_results_csv_path, index=False)
print(f"Predictions saved to {model_results_csv_path}")

# Summary
print("\nSummary of Method to Estimate Player Values:")
print("- **Data Source**: Hard-coded market values based on typical Transfermarkt data for the 2024-2025 season")
print("- **Features Used**:")
print("  - Performance: Gls, Ast, xG, xAG, PrgC, PrgP, PrgR, SCA, GCA, Tkl, Int, Blocks")
print("  - Involvement: Touches, Carries, Rec, Min")
print("  - Metadata: Age, Pos (encoded), Squad (encoded)")
print("- **Model**: XGBoost Regressor")
print("  - Chosen for its ability to handle non-linear relationships and provide feature importance.")
print("  - Hyperparameters tuned using GridSearchCV.")
print("- **Evaluation**:")
print(f"  - MAE: {mae:.2f} million €")
print(f"  - R²: {r2:.2f}")
print("- **Outputs**:")
print(f"  - Actual vs Predicted plot: {os.path.join(plots_dir, 'actual_vs_predicted.png')}")
print(f"  - Feature importance plot: {os.path.join(plots_dir, 'feature_importance.png')}")
print(f"  - Predictions saved to: {model_results_csv_path}")