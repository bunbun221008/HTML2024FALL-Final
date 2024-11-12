#LR with X percent of training data

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import random

# Load the data
train_data = pd.read_csv("train_data.csv")
test_data = pd.read_csv("same_season_test_data.csv")

# Drop string and id columns from train and test data
numeric_train_data = train_data.select_dtypes(exclude=["object"]).drop(columns=["id"])
numeric_test_data = test_data.select_dtypes(exclude=["object"]).drop(columns=["id"])

# Separate features and target variable in the training set
X_train_full = numeric_train_data.drop(columns=["home_team_win"])
y_train_full = numeric_train_data["home_team_win"]

# Handle missing values with mean imputation
X_train_full = X_train_full.fillna(X_train_full.mean())
numeric_test_data = numeric_test_data.fillna(X_train_full.mean())

# Scale the data
scaler = StandardScaler()
X_train_full = scaler.fit_transform(X_train_full)
X_test_full = scaler.transform(numeric_test_data)

# Initialize variables to track the best model, accuracy, and the corresponding X
best_accuracy = 0
best_model = None
best_percent = 0
percentages = range(30, 91, 5)
iterations = 100

# Loop over percentages and iterations
for percent in percentages:
    print(f"Training model on {percent}% of the data...")
    accuracies = []
    for _ in range(iterations):
        # Split the data based on the current percentage
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, train_size=percent / 100.0, random_state=random.randint(0, 10000)
        )
        
        # Train the model
        model = LogisticRegression(max_iter=5000, solver="lbfgs")
        model.fit(X_train, y_train)
        
        # Calculate accuracy on validation set
        val_accuracy = accuracy_score(y_val, model.predict(X_val))
        accuracies.append(val_accuracy)
        
        # If this model has the highest accuracy, save it and track the percentage
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = model
            best_percent = percent

    # Print the average accuracy for the current percentage
    avg_accuracy = np.mean(accuracies)
    print(f"Average accuracy for {percent}% of data: {avg_accuracy:.2f}")

# Use the best model found during the loop to predict on the test set
y_test_pred = best_model.predict(X_test_full)

# Create output DataFrame
output = pd.DataFrame({"id": test_data["id"], "home_team_win": y_test_pred})

# Convert boolean values to True/False strings if needed
output["home_team_win"] = output["home_team_win"].astype(bool)

# Save to CSV
output.to_csv("predictions.csv", index=False)

# Print out best model and corresponding X value
print(f"Best validation accuracy: {best_accuracy:.2f}")
print(f"Best model was trained with {best_percent}% of the data.")
print("Predictions saved to 'predictions.csv'")
