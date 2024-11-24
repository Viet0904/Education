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

# Load the dataset
file_path = "/kaggle/input/data-education/xAPI-Edu-Data.csv"
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


# Define functions for calculating entropy, information gain, intrinsic value, and gain ratio
def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = -np.sum(
        [
            (counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts))
            for i in range(len(elements))
        ]
    )
    return entropy


def info_gain(data, split_attribute_name, target_name="class"):
    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    weighted_entropy = np.sum(
        [
            (counts[i] / np.sum(counts))
            * entropy(
                data.where(data[split_attribute_name] == vals[i]).dropna()[target_name]
            )
            for i in range(len(vals))
        ]
    )
    information_gain = total_entropy - weighted_entropy
    return information_gain


def intrinsic_value(data, split_attribute_name):
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    intrinsic_value = -np.sum(
        [
            (counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts))
            for i in range(len(vals))
        ]
    )
    return intrinsic_value


def gain_ratio(data, split_attribute_name, target_name="class"):
    information_gain = info_gain(data, split_attribute_name, target_name)
    intrinsic_value_score = intrinsic_value(data, split_attribute_name)
    if intrinsic_value_score == 0:
        return 0
    return information_gain / intrinsic_value_score


def rank_features(data, target_name="class"):
    features = data.columns.tolist()
    features.remove(target_name)
    gain_ratios = {}
    for feature in features:
        gain_ratios[feature] = gain_ratio(data, feature, target_name)
    return sorted(gain_ratios.items(), key=lambda x: x[1], reverse=True)


# Initialize lists to store results
all_gain_ratios = []

# Perform ranking and visualization for 5 iterations
n_repeats = 5
for i in range(n_repeats):
    # Split the data
    train_data, _ = train_test_split(data, test_size=0.3, random_state=i)

    # Rank features
    ranked_features = rank_features(train_data, target_name="Class")

    # Extract features and values
    features = [x[0] for x in ranked_features]
    values = [x[1] for x in ranked_features]

    # Append results to the list
    gain_ratios_df = pd.DataFrame({"Feature": features, f"Gain Ratio {i+1}": values})
    all_gain_ratios.append(gain_ratios_df.set_index("Feature"))

    # Plot the rankings
    plt.figure(figsize=(10, 6))
    plt.barh(features, values, color="skyblue")
    plt.xlabel("Gain Ratio")
    plt.ylabel("Attributes")
    plt.title(f"Attribute Ranking by Gain Ratio (Iteration {i + 1})")
    plt.gca().invert_yaxis()
    plt.show()

# Combine results from all iterations
combined_gain_ratios_df = pd.concat(all_gain_ratios, axis=1)
combined_gain_ratios_df.columns = [f"Gain Ratio {i+1}" for i in range(n_repeats)]

# Calculate mean Gain Ratio for each feature
combined_gain_ratios_df["Mean Gain Ratio"] = combined_gain_ratios_df.mean(axis=1)

# Sort by mean Gain Ratio
mean_gain_ratios_df = combined_gain_ratios_df[["Mean Gain Ratio"]].sort_values(
    by="Mean Gain Ratio", ascending=False
)

# Plot the mean Gain Ratio
plt.figure(figsize=(12, 8))
plt.barh(
    mean_gain_ratios_df.index,
    mean_gain_ratios_df["Mean Gain Ratio"],
    color="lightcoral",
)
plt.xlabel("Mean Gain Ratio")
plt.ylabel("Features")
plt.title("Mean Gain Ratio of Features Across 5 Iterations")
plt.gca().invert_yaxis()
plt.grid(True)
plt.show()


# Initialize lists to store results for XGBoost
all_results = []
n_repeats = 5

# Repeat XGBoost procedure n_repeats times
for repeat in range(n_repeats):
    print(f"Repeat {repeat + 1}")
    results = []

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=repeat
    )

    # Run XGBoost with number of features from 1 to 16
    for k in range(1, 17):
        # Get top k features
        top_features = mean_gain_ratios_df.head(k).index

        # Select top features
        X_train_selected = X_train[top_features]
        X_test_selected = X_test[top_features]

        # Initialize XGBoost model
        xgb_model = xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.01, max_depth=6, random_state=repeat
        )

        # Train XGBoost model
        xgb_model.fit(X_train_selected, y_train)

        # Predict on training and testing sets
        y_train_pred = xgb_model.predict(X_train_selected)
        y_test_pred = xgb_model.predict(X_test_selected)

        y_train_pred_prob = xgb_model.predict_proba(X_train_selected)
        y_test_pred_prob = xgb_model.predict_proba(X_test_selected)

        # Compute metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_loss = log_loss(y_train, y_train_pred_prob)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_loss = log_loss(y_test, y_test_pred_prob)

        train_precision = precision_score(y_train, y_train_pred, average="weighted")
        train_recall = recall_score(y_train, y_train_pred, average="weighted")
        train_f1 = f1_score(y_train, y_train_pred, average="weighted")

        test_precision = precision_score(y_test, y_test_pred, average="weighted")
        test_recall = recall_score(y_test, y_test_pred, average="weighted")
        test_f1 = f1_score(y_test, y_test_pred, average="weighted")

        # Store results
        results.append(
            {
                "k": k,
                "Iteration": repeat + 1,
                "Train Accuracy": train_accuracy,
                "Train Loss": train_loss,
                "Test Accuracy": test_accuracy,
                "Test Loss": test_loss,
                "Train Precision": train_precision,
                "Train Recall": train_recall,
                "Train F1-Score": train_f1,
                "Test Precision": test_precision,
                "Test Recall": test_recall,
                "Test F1-Score": test_f1,
            }
        )

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    all_results.append(results_df)

# Combine results from all repeats
combined_results_df = pd.concat(all_results)

# Aggregate metrics across all repeats with mean and standard deviation
final_summary_df = (
    combined_results_df.groupby("k")
    .agg(
        {
            "Train Accuracy": ["mean", "std"],
            "Train Loss": ["mean", "std"],
            "Test Accuracy": ["mean", "std"],
            "Test Loss": ["mean", "std"],
            "Train Precision": ["mean", "std"],
            "Train Recall": ["mean", "std"],
            "Train F1-Score": ["mean", "std"],
            "Test Precision": ["mean", "std"],
            "Test Recall": ["mean", "std"],
            "Test F1-Score": ["mean", "std"],
        }
    )
    .reset_index()
)

# Flatten the multi-index columns
final_summary_df.columns = [
    "k",
    "Train Accuracy Mean",
    "Train Accuracy Std",
    "Train Loss Mean",
    "Train Loss Std",
    "Test Accuracy Mean",
    "Test Accuracy Std",
    "Test Loss Mean",
    "Test Loss Std",
    "Train Precision Mean",
    "Train Precision Std",
    "Train Recall Mean",
    "Train Recall Std",
    "Train F1-Score Mean",
    "Train F1-Score Std",
    "Test Precision Mean",
    "Test Precision Std",
    "Test Recall Mean",
    "Test Recall Std",
    "Test F1-Score Mean",
    "Test F1-Score Std",
]

# Save detailed results (including all iterations) for all repeats
combined_results_df.to_csv("xgb_results_summary_GainRatio_all_repeats.csv", index=False)

# Save summary results with mean and std for each metric
final_summary_df.to_csv("xgb_results_summary_GainRatio_summary.csv", index=False)

# Display final results
print(combined_results_df)
print(final_summary_df)
