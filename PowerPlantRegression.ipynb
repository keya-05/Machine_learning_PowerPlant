# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv('CCPP_data.csv')


# Check for missing values
print("Missing values per column:")
print(data.isnull().sum())

# Drop rows with missing values (if any)
data.dropna(inplace=True)

# EDA
print("Dataset summary statistics:")
print(data.describe())

# Visualize distributions
data.hist(figsize=(10, 8))
plt.suptitle("Feature Distributions")
plt.show()

# Define features (X) and target (y)
X = data[['AT', 'AP', 'RH', 'V']]
y = data['PE']

# Normalize features for Linear Regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize models
lr = LinearRegression()
rf = RandomForestRegressor(random_state=42)

# Train models
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Cross-validation scores
cv_scores_lr = cross_val_score(lr, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_scores_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

print(f"Linear Regression CV RMSE: {np.sqrt(-np.mean(cv_scores_lr))}")
print(f"Random Forest CV RMSE: {np.sqrt(-np.mean(cv_scores_rf))}")

# Make predictions
y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)

# Evaluate models
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_lr = r2_score(y_test, y_pred_lr)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Linear Regression RMSE: {rmse_lr}")
print(f"Random Forest RMSE: {rmse_rf}")
print(f"Linear Regression R²: {r2_lr}")
print(f"Random Forest R²: {r2_rf}")

# Feature Importance (Random Forest)
feature_importance = rf.feature_importances_
feature_names = X.columns
plt.barh(feature_names, feature_importance)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance from Random Forest')
plt.show()

# Visualization: Actual vs Predicted
plt.scatter(y_test, y_pred_lr, label='Linear Regression', alpha=0.6)
plt.scatter(y_test, y_pred_rf, label='Random Forest', alpha=0.6, color='orange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Comparison')
plt.legend()
plt.show()

# Residual plot for Random Forest
residuals = y_test - y_pred_rf
plt.scatter(y_test, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Actual')
plt.ylabel('Residuals')
plt.title('Residual Plot for Random Forest')
plt.show()
