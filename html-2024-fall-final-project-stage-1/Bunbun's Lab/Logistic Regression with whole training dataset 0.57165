import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

# Step 1: Load the training and testing data
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('same_season_test_data.csv')

# Step 2: Select only numeric columns (excluding non-numeric)
X_train = train_data.select_dtypes(include=['number'])  # Select numeric columns only
y_train = train_data['home_team_win']  # Target column from train data

X_test = test_data.select_dtypes(include=['number'])  # Select numeric columns only

# Check for missing values in the training and test data
print(f"Missing values in train data: \n{X_train.isnull().sum()}")
print(f"Missing values in test data: \n{X_test.isnull().sum()}")

# Step 3: Impute missing values (fill with column mean)
imputer = SimpleImputer(strategy='mean')  # Impute using mean of each column
X_train_imputed = imputer.fit_transform(X_train)  # Fit and transform the training data
X_test_imputed = imputer.transform(X_test)  # Transform the test data with the same mean values

# Train the logistic regression model
model = LogisticRegression(max_iter=10000)  # Set higher max_iter for convergence
model.fit(X_train_imputed, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test_imputed)

# Step 4: Create a DataFrame with the predictions
output_df = pd.DataFrame({
    'id': test_data.index,  # Assuming the index is the ID for the test data
    'home_team_win': y_pred.astype(bool)  # Convert to boolean
})

# Step 5: Save the predictions to a CSV file
output_df.to_csv('predictions.csv', index=False)

print("Predictions saved to 'predictions.csv'.")
