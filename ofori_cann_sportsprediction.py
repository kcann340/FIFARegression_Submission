# -*- coding: utf-8 -*-
"""Ofori-Cann_SportsPrediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1lVezIWFi9YayHjjaWKQtDdZgQTOYmP2T
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import pickle

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load data
playertrain_df = pd.read_csv("/content/male_players (legacy).csv") # For training
playertest_df = pd.read_csv("/content/players_22 (1).csv") # For testing

# Display initial data
print(playertrain_df.head())
print(playertest_df.head())
print(playertrain_df.info())
print(playertest_df.info())

# Check for missing values
print("Checking sum of missing value for Players 21 (Train Data):")
print(playertrain_df.isnull().sum())
print("Checking sum of missing value for Players 22 (Test Data):")
print(playertest_df.isnull().sum())

# Drop columns with more than 30% missing values
total_rows_21 = playertrain_df.shape[0]
threshold_21 = int(0.3 * total_rows_21)
columns_to_drop = [col for col in playertrain_df.columns if playertrain_df[col].isna().sum() > threshold_21]

playertrain_df = playertrain_df.drop(columns=columns_to_drop)
playertest_df = playertest_df.drop(columns=columns_to_drop)

# Further drop specific columns
drop_columns = ['player_url','long_name','dob','body_type','real_face','player_face_url']
playertrain_df = playertrain_df.drop(drop_columns, axis=1)
playertest_df = playertest_df.drop(drop_columns, axis=1)

# Drop additional columns
drop_r_cols = ['short_name', 'player_positions', 'league_name', 'nationality_name', 'ls', 'st', 'rs', 'lw', 'lf', 'cf', 'rf', 'rw', 'lam', 'cam', 'ram',
               'lm', 'lcm', 'cm', 'rcm', 'rm', 'lwb', 'ldm', 'cdm', 'rdm', 'rwb', 'lb', 'lcb', 'cb', 'rcb', 'rb', 'gk']
playertrain_df = playertrain_df.drop(drop_r_cols, axis=1)
playertest_df = playertest_df.drop(drop_r_cols, axis=1)

# Align the columns of the training and test datasets
playertest_df = playertest_df.reindex(columns=playertrain_df.columns, fill_value=np.nan)

# Impute missing values
num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')

num_features = playertrain_df.select_dtypes(include=[np.number]).columns.tolist()
cat_features = playertrain_df.select_dtypes(include=[object]).columns.tolist()

# Remove the target variable from the feature list
num_features.remove('overall')

# Impute numerical and categorical data for training data
playertrain_df[num_features] = num_imputer.fit_transform(playertrain_df[num_features])
playertrain_df[cat_features] = cat_imputer.fit_transform(playertrain_df[cat_features])

# Impute numerical and categorical data for testing data
playertest_df[num_features] = num_imputer.transform(playertest_df[num_features])
playertest_df[cat_features] = cat_imputer.transform(playertest_df[cat_features])

# One-hot encode categorical features
playertrain_encoded_df = pd.get_dummies(playertrain_df, columns=cat_features, drop_first=True)
playertest_encoded_df = pd.get_dummies(playertest_df, columns=cat_features, drop_first=True)

# Align test set to training set
playertest_encoded_df = playertest_encoded_df.reindex(columns=playertrain_encoded_df.columns, fill_value=0)

# The target variable and features
X = playertrain_encoded_df.drop(columns=['overall'])
y = playertrain_encoded_df['overall']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=69)

# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Saving the scaler
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

def train_model(model, param_grid, X, y):
    cv = KFold(n_splits=7, random_state=69, shuffle=True)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X, y)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score (MAE): {-grid_search.best_score_}")
    return grid_search.best_estimator_

print("\nTraining XGBoost...")
xgb_model = xgb.XGBRegressor(random_state=42)
xgb_params = {
    'n_estimators': [100, 500, 1000],
    'learning_rate': [0.1, 0.001, 0.01],
    'max_depth': [3, 5, 9, 15],
    'colsample_bytree': [0.5, 0.75, 1]
}
best_xgb = train_model(xgb_model, xgb_params, X_train_scaled, y_train)

print("\nTraining Gradient Boosting...")
gbr_model = GradientBoostingRegressor(random_state=63)
gbr_params = {
    'n_estimators': [100, 500, 1000],
    'learning_rate': [0.1, 0.001, 0.01],
    'max_depth': [9, 15]
}
best_gbr = train_model(gbr_model, gbr_params, X_train_scaled, y_train)

print("\nTraining Random Forest...")
rf_model = RandomForestRegressor(random_state=39)
rf_params = {
    'n_estimators': [500, 1000],
    'max_depth': [12, 15],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
best_rf = train_model(rf_model, rf_params, X_train_scaled, y_train)

# Create an ensemble model
ensemble = VotingRegressor(
    estimators=[
        ('xgb', best_xgb),
        ('gbr', best_gbr),
        ('rf', best_rf)
    ]
)

# Fit model on the training data
print("\nTraining Ensemble Model...")
ensemble.fit(X_train_scaled, y_train)

# Predict and evaluate on the training set
train_pred = ensemble.predict(X_train_scaled)
train_mae = mean_absolute_error(y_train, train_pred)
print(f"Ensemble model MAE on training set: {train_mae}")

# Save models
with open('best_xgb_model.pkl', 'wb') as file:
    pickle.dump(best_xgb, file)

with open('best_gbr_model.pkl', 'wb') as file:
    pickle.dump(best_gbr, file)

with open('best_rf_model.pkl', 'wb') as file:
    pickle.dump(best_rf, file)

with open('ensemble_model.pkl', 'wb') as file:
    pickle.dump(ensemble, file)

def evaluate_model_on_test(model, X_test, y_test, model_name):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"{model_name} model MAE on test set: {mae:.2f}")

print("\nEvaluating XGBoost...")
evaluate_model_on_test(best_xgb, X_test_scaled, y_test, "XGBoost")

print("\nEvaluating Gradient Boosting...")
evaluate_model_on_test(best_gbr, X_test_scaled, y_test, "Gradient Boosting")

print("\nEvaluating Random Forest...")
evaluate_model_on_test(best_rf, X_test_scaled, y_test, "Random Forest")

print("\nEvaluating Ensemble...")
evaluate_model_on_test(ensemble, X_test_scaled, y_test, "Ensemble")

# Evaluate models on Players 22 dataset
X_22 = playertest_encoded_df.drop(columns=['overall'])
y_22 = playertest_encoded_df['overall']
X_scaled_22 = scaler.transform(X_22)

# Load models
with open('best_xgb_model.pkl', 'rb') as file:
    lbest_xgb = pickle.load(file)

with open('best_gbr_model.pkl', 'rb') as file:
    lbest_gbr = pickle.load(file)

with open('best_rf_model.pkl', 'rb') as file:
    lbest_rf = pickle.load(file)

with open('ensemble_model.pkl', 'rb') as file:
    lensemble = pickle.load(file)

print("\nEvaluating XGBoost on Players 22...")
evaluate_model_on_test(lbest_xgb, X_scaled_22, y_22, "XGBoost")

print("\nEvaluating Gradient Boosting on Players 22...")
evaluate_model_on_test(lbest_gbr, X_scaled_22, y_22, "Gradient Boosting")

print("\nEvaluating Random Forest on Players 22...")
evaluate_model_on_test(lbest_rf, X_scaled_22, y_22, "Random Forest")

print("\nEvaluating Ensemble on Players 22...")
evaluate_model_on_test(lensemble, X_scaled_22, y_22, "Ensemble")

!pip freeze > requirements.txt






