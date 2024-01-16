# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 22:36:00 2024

@author: andrei.nichita0308
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# Load the dataset
column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']
df = pd.read_csv('P1-cars.csv', delim_whitespace=True, names=column_names)

# Handling missing values
# Convert '?' to NaN
df['horsepower'].replace('?', pd.NA, inplace=True)

# Convert 'horsepower' column to numeric
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')

# Impute missing values using the mean
imputer = SimpleImputer(strategy='mean')
df['horsepower'] = imputer.fit_transform(df[['horsepower']])

# Split the data into features (X) and target variable (y)
X = df.drop(['mpg', 'car_name'], axis=1)
y = df['mpg']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Build and evaluate a Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)

linear_mse = mean_squared_error(y_test, linear_predictions)
print(f'Linear Regression Mean Squared Error: {linear_mse}')

linear_accuracy = r2_score(y_test, linear_predictions)
print(f"Linear Regression Accuracy: {linear_accuracy:.4f}")
print("\n")



# Build and evaluate a Random Forest Regressor model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

rf_mse = mean_squared_error(y_test, rf_predictions)
print(f'Random Forest Regressor Mean Squared Error: {rf_mse}')

rf_accuracy = r2_score(y_test, rf_predictions)
print(f"Random Forest Regression Accuracy: {rf_accuracy:.4f}")
print("\n")


# Build and evaluate a Support Vector Regressor model
svr_model = make_pipeline(SimpleImputer(strategy='mean'), SVR())
svr_model.fit(X_train, y_train)
svr_predictions = svr_model.predict(X_test)

svr_mse = mean_squared_error(y_test, svr_predictions)
print(f'Support Vector Regressor Mean Squared Error: {svr_mse}')

svr_accuracy = r2_score(y_test, svr_predictions)
print(f"SVR Accuracy: {svr_accuracy:.4f}")
print("\n")


# Model names and corresponding accuracies
models = ['Linear Regression', 'Random Forest Regression', 'SVR']
accuracies = [linear_accuracy, rf_accuracy, svr_accuracy]

# Plotting the accuracies obtained
plt.bar(models, accuracies, color=['blue', 'green', 'red'])
plt.xlabel('Regression Models')
plt.ylabel('R-squared (Accuracy)')
plt.title('Accuracy of Regression Models')
plt.ylim(0, 1)  # Setting y-axis limit between 0 and 1 for R-squared values
plt.show()




# Plotting the differences
plt.figure(figsize=(12, 6))

# Plot for Linear Regression
plt.subplot(1, 3, 1)
plt.scatter(y_test, linear_predictions, color='blue', alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2)
plt.title('Linear Regression - Actual vs Predicted')
plt.xlabel('Actual MPG')
plt.ylabel('Predicted MPG')

# Plot for Random Forest Regressor
plt.subplot(1, 3, 2)
plt.scatter(y_test, rf_predictions, color='green', alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2)
plt.title('Random Forest Regressor - Actual vs Predicted')
plt.xlabel('Actual MPG')
plt.ylabel('Predicted MPG')

# Plot for Support Vector Regressor
plt.subplot(1, 3, 3)
plt.scatter(y_test, svr_predictions, color='orange', alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2)
plt.title('Support Vector Regressor - Actual vs Predicted')
plt.xlabel('Actual MPG')
plt.ylabel('Predicted MPG')

plt.tight_layout()
plt.show()
