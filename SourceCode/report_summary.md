# Player Value Estimation Report
## Data Collection
- Used hard-coded transfer values for 50 players due to inability to access https://www.footballtransfers.com.
- Matched 45 players using fuzzy matching (threshold=75–80).
## Feature Selection
- Numeric: Age, Min, Gls, Ast, xG, xAG, PrgC_standard, PrgP_standard, PrgR_standard, Tkl, Int, Blocks, Touches, Carries, Rec, Won\%, Save\%, CS\%, Gls_Ast.\n- Categorical: Pos, Squad (one-hot encoded).
- Rationale: Features capture performance, expected contributions, progression, involvement, and defensive actions relevant to market value.
- Missing features: None.
## Model
- XGBoost Regressor with parameters: {'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 100, 'subsample': 0.8}.
- Training MAE: 16.85M €, Test MAE: 20.74M €, Test R²: -0.01.
- Cross-Validation MAE: 22.78 ± 3.92M €.
- Low R² (-0.01) and high MAE (20.74M €) indicate the model is unreliable, likely due to the small dataset (45/307).
## Results
- Actual vs Predicted: See `actual_vs_predicted.png`.
- Feature Importance: See `feature_importance.png`.
- Predictions: See C:\Users\nguye\Downloads\player_value_predictions.csv.
## Limitations
- Small dataset (45/307 players) due to limited hard-coded data.
- Low R² (-0.01) and high MAE (20.74M €) indicate poor predictive power.
- 5 unmatched players (Rodri, Jack Grealish, John Stones, Julian Alvarez, Richarlison) due to absence in results.csv (likely <900 minutes).
