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
from catboost import CatBoostClassifier

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


from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

# Define parameter grid for CatBoost
param_grid = {
    "iterations": [100, 200, 500, 1000],  # Số vòng lặp
    "depth": [3, 4, 5, 6, 7, 8, 9, 10],  # Độ sâu của cây quyết định
    "learning_rate": [0.01, 0.1, 0.2, 0.02, 0.3, 0.03],  # Tốc độ học
    "l2_leaf_reg": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Hệ số điều chuẩn L2 cho lá
    "leaf_estimation_iterations": [1, 3, 4, 5, 7, 9, 10],
    "loss_function": ["Logloss", "MultiClass"],
}

# Create a CatBoostClassifier instance
ct = CatBoostClassifier(logging_level="Silent")

# Use GridSearchCV to find the best parameters
grid_search = GridSearchCV(estimator=ct, param_grid=param_grid, cv=5)
grid_search.fit(
    X_train, y_train, early_stopping_rounds=10, use_best_model=False, verbose=False
)


# Check results
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

# Use the best model to predict and evaluate
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))

# Get the best parameters
best_params = grid_search.best_params_


# Create a filename with the algorithm name and current date-time
filename = f"CatBoostClassifier1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# Save the best parameters to the file
with open(filename, "w") as file:
    file.write(f"Best set of hyperparameters: {best_params}\n")
    file.write(f"Best score: {grid_search.best_score_}\n")

print(f"Best hyperparameters and score saved to {filename}")
