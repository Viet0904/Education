import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import shutil

# from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from datetime import datetime

# Load the dataset
file_path = "xAPI-Edu-Data.csv"
data = pd.read_csv(file_path)

label_encoders = {}
categorical_columns = [
    "gender",
    "NationalITy",
    "PlaceofBirth",
    "StageID",
    "GradeID",
    "SectionID",
    "Topic",
    "Semester",
    "Relation",
    "ParentAnsweringSurvey",
    "ParentschoolSatisfaction",
    "StudentAbsenceDays",
    "Class",
]

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Define features and target
X = data.drop("Class", axis=1)
y = data["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# Define the hyperparameter grid
param_grid = {
    "booster": ["gbtree ", "gblinear "],
    "eta ": [0.3, 0.2, 0.1],
    "max_depth": [3, 4, 5, 6, 7, 8, 9, 10],
    "learning_rate": [0.1, 0.01, 0.001, 0.05, 0.02, 0.2, 0.03, 0.3, 0.04, 0.4, 0.05],
    "min_child_weight": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "gamma": [0, 0.5, 1, 1.5, 2, 5],
    "subsample": [0.6, 0.8, 1.0, 0.9, 0.7, 0.5, 1],
    "n_estimators": [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
}


# Create the XGBoost model object with early stopping in the constructor
xgb_model = xgb.XGBClassifier()

# Split the training data into training and validation sets
X_train_part, X_val, y_train_part, y_val = train_test_split(
    X_train, y_train, test_size=0.3, random_state=42
)

# Create the GridSearchCV object
grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring="accuracy")

# Fit the GridSearchCV object to the training data with early stopping
grid_search.fit(
    X_train_part,
    y_train_part,
    eval_set=[(X_val, y_val)],  # Provide validation set for early stopping
    verbose=False,
)

# Print the best set of hyperparameters and the corresponding score
print("Best set of hyperparameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)
# Get the best parameters
best_params = grid_search.best_params_

# Create a filename with the algorithm name and current date-time
filename = f"XGBoost4_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# Save the best parameters to the file
with open(filename, "w") as file:
    file.write(f"Best set of hyperparameters: {best_params}\n")
    file.write(f"Best score: {grid_search.best_score_}\n")

print(f"Best hyperparameters and score saved to {filename}")
