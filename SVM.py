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

param_grid = {
    "C": [0.1, 1, 10, 50, 100],
    "gamma": [1, 0.1, 0.01, 0.001],
    "kernel": ["linear", "rbf", "poly"],
    "degree": [2, 3, 4],
    "tol": [1e-3, 1e-4, 1e-2, 1e-5, 1e-6],
}

grid_search = GridSearchCV(SVC(), param_grid, refit=True, cv=5)

# Huấn luyện mô hình
grid_search.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
predictions = grid_search.predict(X_test)

# In ra báo cáo kết quả
print("Best parameters found: ", grid_search.best_params_)
print("Classification report:\n", classification_report(y_test, predictions))
# Get the best parameters
best_params = grid_search.best_params_


# Create a filename with the algorithm name and current date-time
filename = f"SVM1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# Save the best parameters to the file
with open(filename, "w") as file:
    file.write(f"Best set of hyperparameters: {best_params}\n")
    file.write(f"Best score: {grid_search.best_score_}\n")

print(f"Best hyperparameters and score saved to {filename}")
