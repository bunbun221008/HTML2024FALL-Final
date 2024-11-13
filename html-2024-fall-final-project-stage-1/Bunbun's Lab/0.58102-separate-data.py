#separate training data to 4 parts. trained a LR model on each part of data. 
#finally use trainable weighted grometric average to calculate the final predicted wining probability. >0.5 = TRUE otherwise FALSE

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss, accuracy_score
from scipy.optimize import minimize


# Load your data
data = pd.read_csv("train_data.csv")  # Update this to the actual data path if needed
test_data = pd.read_csv("same_season_test_data.csv")

# Separate data into the four parts based on assumed feature names
# Note: Adjust 'recent_batting_features', 'recent_pitching_features', etc. with actual column names
target_column = 'home_team_win' 
feature_sets = {
    "recent_batting_features" : ["home_team_rest","away_team_rest","home_pitcher_rest","away_pitcher_rest","home_team_errors_mean","home_team_errors_std","home_team_errors_skew","away_team_errors_mean","away_team_errors_std","away_team_errors_skew","home_team_spread_mean","home_team_spread_std","home_team_spread_skew","away_team_spread_mean","away_team_spread_std","away_team_spread_skew","home_team_wins_mean","home_team_wins_std","home_team_wins_skew","away_team_wins_mean","away_team_wins_std","away_team_wins_skew","home_batting_batting_avg_10RA","home_batting_onbase_perc_10RA","home_batting_onbase_plus_slugging_10RA","home_batting_leverage_index_avg_10RA","home_batting_RBI_10RA","away_batting_batting_avg_10RA","away_batting_onbase_perc_10RA","away_batting_onbase_plus_slugging_10RA","away_batting_leverage_index_avg_10RA","away_batting_RBI_10RA"],  # placeholder
    "recent_pitching_features" : ["home_team_rest","away_team_rest","home_pitcher_rest","away_pitcher_rest","home_team_errors_mean","home_team_errors_std","home_team_errors_skew","away_team_errors_mean","away_team_errors_std","away_team_errors_skew","home_team_spread_mean","home_team_spread_std","home_team_spread_skew","away_team_spread_mean","away_team_spread_std","away_team_spread_skew","home_team_wins_mean","home_team_wins_std","home_team_wins_skew","away_team_wins_mean","away_team_wins_std","away_team_wins_skew","home_pitching_earned_run_avg_10RA","home_pitching_SO_batters_faced_10RA","home_pitching_H_batters_faced_10RA","home_pitching_BB_batters_faced_10RA","away_pitching_earned_run_avg_10RA","away_pitching_SO_batters_faced_10RA","away_pitching_H_batters_faced_10RA","away_pitching_BB_batters_faced_10RA","home_pitcher_earned_run_avg_10RA","home_pitcher_SO_batters_faced_10RA","home_pitcher_H_batters_faced_10RA","home_pitcher_BB_batters_faced_10RA","away_pitcher_earned_run_avg_10RA","away_pitcher_SO_batters_faced_10RA","away_pitcher_H_batters_faced_10RA","away_pitcher_BB_batters_faced_10RA"],  # placeholder
    "seasonal_batting_features" : ["home_team_rest","away_team_rest","home_pitcher_rest","away_pitcher_rest","home_team_errors_mean","home_team_errors_std","home_team_errors_skew","away_team_errors_mean","away_team_errors_std","away_team_errors_skew","home_team_spread_mean","home_team_spread_std","home_team_spread_skew","away_team_spread_mean","away_team_spread_std","away_team_spread_skew","home_team_wins_mean","home_team_wins_std","home_team_wins_skew","away_team_wins_mean","away_team_wins_std","away_team_wins_skew","home_batting_batting_avg_mean","home_batting_batting_avg_std","home_batting_batting_avg_skew","home_batting_onbase_perc_mean","home_batting_onbase_perc_std","home_batting_onbase_perc_skew","home_batting_onbase_plus_slugging_mean","home_batting_onbase_plus_slugging_std","home_batting_onbase_plus_slugging_skew","home_batting_leverage_index_avg_mean","home_batting_leverage_index_avg_std","home_batting_leverage_index_avg_skew","home_batting_wpa_bat_mean","home_batting_wpa_bat_std","home_batting_wpa_bat_skew","home_batting_RBI_mean","home_batting_RBI_std","home_batting_RBI_skew","away_batting_batting_avg_mean","away_batting_batting_avg_std","away_batting_batting_avg_skew","away_batting_onbase_perc_mean","away_batting_onbase_perc_std","away_batting_onbase_perc_skew","away_batting_onbase_plus_slugging_mean","away_batting_onbase_plus_slugging_std","away_batting_onbase_plus_slugging_skew","away_batting_leverage_index_avg_mean","away_batting_leverage_index_avg_std","away_batting_leverage_index_avg_skew","away_batting_wpa_bat_mean","away_batting_wpa_bat_std","away_batting_wpa_bat_skew","away_batting_RBI_mean","away_batting_RBI_std","away_batting_RBI_skew"], # placeholder
    "seasonal_pitching_features" : ["home_team_rest","away_team_rest","home_pitcher_rest","away_pitcher_rest","home_team_errors_mean","home_team_errors_std","home_team_errors_skew","away_team_errors_mean","away_team_errors_std","away_team_errors_skew","home_team_spread_mean","home_team_spread_std","home_team_spread_skew","away_team_spread_mean","away_team_spread_std","away_team_spread_skew","home_team_wins_mean","home_team_wins_std","home_team_wins_skew","away_team_wins_mean","away_team_wins_std","away_team_wins_skew","home_pitching_earned_run_avg_mean","home_pitching_earned_run_avg_std","home_pitching_earned_run_avg_skew","home_pitching_SO_batters_faced_mean","home_pitching_SO_batters_faced_std","home_pitching_SO_batters_faced_skew","home_pitching_H_batters_faced_mean","home_pitching_H_batters_faced_std","home_pitching_H_batters_faced_skew","home_pitching_BB_batters_faced_mean","home_pitching_BB_batters_faced_std","home_pitching_BB_batters_faced_skew","home_pitching_leverage_index_avg_mean","home_pitching_leverage_index_avg_std","home_pitching_leverage_index_avg_skew","home_pitching_wpa_def_mean","home_pitching_wpa_def_std","home_pitching_wpa_def_skew","away_pitching_earned_run_avg_mean","away_pitching_earned_run_avg_std","away_pitching_earned_run_avg_skew","away_pitching_SO_batters_faced_mean","away_pitching_SO_batters_faced_std","away_pitching_SO_batters_faced_skew","away_pitching_H_batters_faced_mean","away_pitching_H_batters_faced_std","away_pitching_H_batters_faced_skew","away_pitching_BB_batters_faced_mean","away_pitching_BB_batters_faced_std","away_pitching_BB_batters_faced_skew","away_pitching_leverage_index_avg_mean","away_pitching_leverage_index_avg_std","away_pitching_leverage_index_avg_skew","away_pitching_wpa_def_mean","away_pitching_wpa_def_std","away_pitching_wpa_def_skew","home_pitcher_earned_run_avg_mean","home_pitcher_earned_run_avg_std","home_pitcher_earned_run_avg_skew","home_pitcher_SO_batters_faced_mean","home_pitcher_SO_batters_faced_std","home_pitcher_SO_batters_faced_skew","home_pitcher_H_batters_faced_mean","home_pitcher_H_batters_faced_std","home_pitcher_H_batters_faced_skew","home_pitcher_BB_batters_faced_mean","home_pitcher_BB_batters_faced_std","home_pitcher_BB_batters_faced_skew","home_pitcher_leverage_index_avg_mean","home_pitcher_leverage_index_avg_std","home_pitcher_leverage_index_avg_skew","home_pitcher_wpa_def_mean","home_pitcher_wpa_def_std","home_pitcher_wpa_def_skew","away_pitcher_earned_run_avg_mean","away_pitcher_earned_run_avg_std","away_pitcher_earned_run_avg_skew","away_pitcher_SO_batters_faced_mean","away_pitcher_SO_batters_faced_std","away_pitcher_SO_batters_faced_skew","away_pitcher_H_batters_faced_mean","away_pitcher_H_batters_faced_std","away_pitcher_H_batters_faced_skew","away_pitcher_BB_batters_faced_mean","away_pitcher_BB_batters_faced_std","away_pitcher_BB_batters_faced_skew","away_pitcher_leverage_index_avg_mean","away_pitcher_leverage_index_avg_std","away_pitcher_leverage_index_avg_skew","away_pitcher_wpa_def_mean","away_pitcher_wpa_def_std","away_pitcher_wpa_def_skew"]  # placeholder
}
# Validate features
for feature_set_name, features in feature_sets.items():
    missing_features = [f for f in features if f not in data.columns]
    if missing_features:
        raise ValueError(f"Missing features in {feature_set_name}: {missing_features}")
if target_column not in data.columns:
    raise ValueError(f"Missing target column: {target_column}")



# Create pipelines for base models
base_models = {
    "model_1": make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(random_state=42)),
    "model_2": make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(random_state=42)),
    "model_3": make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(random_state=42)),
    "model_4": make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(random_state=42))
}

# Map model names to their corresponding feature subsets
model_to_features = {
    "model_1": "recent_batting_features",
    "model_2": "recent_pitching_features",
    "model_3": "seasonal_batting_features",
    "model_4": "seasonal_pitching_features"
}

# Repeat the process 100 times to find the best model
n_repeats = 100
best_model_info = {"accuracy": 0, "weights": None, "random_state": None}

for i in range(n_repeats):
    print(f"Repeat {i + 1}/{n_repeats}")
    # Split data into training, validation, and testing sets
    X_train_full, X_val, y_train_full, y_val = train_test_split(data, data[target_column], test_size=0.2, random_state=i)

    # Train base models and collect probabilities for validation
    val_probabilities = []

    for model_name, pipeline in base_models.items():
        # Retrieve the corresponding feature subset
        feature_subset_name = model_to_features[model_name]
        feature_subset = feature_sets[feature_subset_name]
        X_train_full_subset = X_train_full[feature_subset]
        X_val_subset = X_val[feature_subset]
        
        # Train the model
        pipeline.fit(X_train_full_subset, y_train_full)
        
        # Get probabilities for validation
        val_prob = pipeline.predict_proba(X_val_subset)[:, 1]
        val_probabilities.append(val_prob)

    # Combine probabilities into an array
    val_probabilities = np.array(val_probabilities)

    # Define the optimization function
    def objective(weights):
        weighted_probs = np.exp(np.sum(weights[:, np.newaxis] * np.log(val_probabilities), axis=0))
        return log_loss(y_val, weighted_probs)

    # Initial weights
    initial_weights = np.ones(len(base_models)) / len(base_models)

    # Constraints: weights must sum to 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

    # Bounds: weights must be between 0 and 1
    bounds = [(0, 1) for _ in range(len(base_models))]

    # Optimize weights
    result = minimize(objective, initial_weights, bounds=bounds, constraints=constraints)
    optimal_weights = result.x

    # Calculate validation accuracy using the optimal weights
    weighted_probs_val = np.exp(np.sum(optimal_weights[:, np.newaxis] * np.log(val_probabilities), axis=0))
    val_predictions = (weighted_probs_val >= 0.5).astype(int)
    accuracy = accuracy_score(y_val, val_predictions)

    # Update the best model information if necessary
    if accuracy > best_model_info["accuracy"]:
        best_model_info["accuracy"] = accuracy
        best_model_info["weights"] = optimal_weights
        best_model_info["random_state"] = i

# Train the best model using the best random state
best_random_state = best_model_info["random_state"]
X_train_full, X_val, y_train_full, y_val = train_test_split(data, data[target_column], test_size=0.2, random_state=best_random_state)

# Train base models and collect probabilities for the test data
test_probabilities = []

for model_name, pipeline in base_models.items():
    # Retrieve the corresponding feature subset
    feature_subset_name = model_to_features[model_name]
    feature_subset = feature_sets[feature_subset_name]
    X_train_full_subset = X_train_full[feature_subset]
    X_test_subset = test_data[feature_subset]
    
    # Train the model
    pipeline.fit(X_train_full_subset, y_train_full)
    
    # Get probabilities for the test data
    test_prob = pipeline.predict_proba(X_test_subset)[:, 1]
    test_probabilities.append(test_prob)

# Combine test probabilities into an array
test_probabilities = np.array(test_probabilities)

# Apply the best weights to the test probabilities
weighted_probs_test = np.exp(np.sum(best_model_info["weights"][:, np.newaxis] * np.log(test_probabilities), axis=0))

# Final predictions for the test data
final_predictions = (weighted_probs_test >= 0.5).astype(bool)  # Convert to True/False

# Prepare predictions in the specified format
predictions_df = pd.DataFrame({
    "id": range(len(final_predictions)),
    "home_team_win": final_predictions
})

# Save predictions to CSV
predictions_df.to_csv("predictions.csv", index=False)
print(f"Predictions saved to predictions.csv")
print(f"Best accuracy: {best_model_info['accuracy']:.4f}")
