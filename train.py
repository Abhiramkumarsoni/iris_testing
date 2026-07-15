# importing libraries
import os
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris
#loadging prebuilt iris dataset from sklearn

iris = load_iris()
X = iris.data
y = iris.target

# splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# let's use random forest regressor to train the model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train,y_train)

# making predictions on the test set
y_pred = model.predict(X_test)

#printing the feature that we have used to train the model
print(f"Features used for training: {iris.feature_names}")

# evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")
print(f"Root Mean Squared Error: {rmse}")


# saving the model to the dist using the joblib library
import joblib
model_filename="random_forest_model.pkl"
joblib.dump(model, model_filename)

