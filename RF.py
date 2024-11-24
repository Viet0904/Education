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

# Xác định các tham số cần tối ưu trong RandomForest
param_grid = {
    "n_estimators": [50, 100, 150, 250, 300, 500],  # Số lượng cây trong rừng
    "max_depth": [None, 5, 10, 20, 30, 50, 100],  # Chiều sâu tối đa của mỗi cây
    "min_samples_split": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
    ],  # Số lượng mẫu tối thiểu để tách một node
    "min_samples_leaf": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
    ],  # Số lượng mẫu tối thiểu trong một node lá
    "bootstrap": [True, False],  # Có sử dụng bootstrap mẫu hay không
    "criterion": ["gini", "entropy", "log_loss"],
    "max_features": [None, "sqrt", "log2"],
    "class_weight": ["balanced", "balanced_subsample"],
}

# Sử dụng GridSearchCV để tìm kiếm tham số tốt nhất
rf = RandomForestClassifier()
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)  # Perform the search

# Kiểm tra kết quả
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

# Dùng mô hình với tham số tốt nhất để dự đoán và đánh giá
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
# Get the best parameters
best_params = grid_search.best_params_


# Create a filename with the algorithm name and current date-time
filename = f"RF1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# Save the best parameters to the file
with open(filename, "w") as file:
    file.write(f"Best set of hyperparameters: {best_params}\n")
    file.write(f"Best score: {grid_search.best_score_}\n")

print(f"Best hyperparameters and score saved to {filename}")
