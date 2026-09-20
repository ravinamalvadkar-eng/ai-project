import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    roc_auc_score, 
    classification_report, 
    confusion_matrix
)

# ==========================================
# 1. LOAD DATASET
# ==========================================
# Load Telco Customer Churn dataset (Kaggle/IBM Telco Dataset)
# Replace 'WA_Fn-UseC_-Telco-Customer-Churn.csv' with your local dataset path
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# ==========================================
# 2. DATA PREPROCESSING & CLEANING (PDF Sec 3)
# ==========================================
# Handle blank spaces in 'TotalCharges' and convert to float
df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

# Impute missing TotalCharges with median
total_charges_median = df['TotalCharges'].median()
df['TotalCharges'].fillna(total_charges_median, inplace=True)

# Encode Target Variable 'Churn' (Yes -> 1, No -> 0)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Drop customerID as it is a unique identifier
if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])

# Identify Feature Types
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = [col for col in df.columns if col not in numerical_features + ['Churn']]

X = df.drop(columns=['Churn'])
y = df['Churn']

# Define Preprocessing Pipelines using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ]
)

# ==========================================
# 3. EXPLORATORY DATA ANALYSIS (EDA) (PDF Sec 4)
# ==========================================
# Figure 1: Churn Distribution by Contract Type
plt.figure(figsize=(8, 5))
contract_churn = df.groupby('Contract')['Churn'].mean() * 100
ax = contract_churn.plot(kind='bar', color=['#f44336', '#2196f3', '#4caf50'])
plt.title('Figure 1: Churn Distribution by Contract Type (%)', fontsize=12)
plt.xlabel('Contract Type')
plt.ylabel('Churn Rate (%)')
plt.xticks(rotation=0)

for p in ax.patches:
    ax.annotate(f"{p.get_height():.1f}%", 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')

plt.tight_layout()
plt.show()

# ==========================================
# 4. MODEL BUILDING & SPLITTING (PDF Sec 5)
# ==========================================
# 80/20 Stratified Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# Create Pipelines for both models
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
])

# Train Models
lr_pipeline.fit(X_train, y_train)
rf_pipeline.fit(X_train, y_train)

# ==========================================
# 5. MODEL EVALUATION (PDF Sec 6)
# ==========================================
def evaluate_model(model, name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"=== {name} Performance ===")
    print(f"Accuracy:  {acc * 100:.1f}%")
    print(f"Precision: {prec * 100:.1f}%")
    print(f"Recall:    {rec * 100:.1f}%")
    print(f"ROC-AUC:   {auc:.2f}\n")
    return {'Model': name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'ROC-AUC': auc}

print("\n--- MODEL PERFORMANCE EVALUATION ---")
lr_metrics = evaluate_model(lr_pipeline, "Logistic Regression")
rf_metrics = evaluate_model(rf_pipeline, "Random Forest")

# Figure 2: Model Performance Comparison
results_df = pd.DataFrame([lr_metrics, rf_metrics])
plt.figure(figsize=(8, 4))
bars = plt.barh(results_df['Model'], results_df['Accuracy'] * 100, color=['#9e9e9e', '#2196f3'])
plt.title('Figure 2: Model Performance Comparison (Accuracy)', fontsize=12)
plt.xlabel('Accuracy (%)')
plt.xlim(0, 100)

for bar in bars:
    width = bar.get_width()
    plt.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', va='center')

plt.tight_layout()
plt.show()

# ==========================================
# 6. FEATURE IMPORTANCE ANALYSIS (PDF Sec 7)
# ==========================================
# Extract Feature Importances from Random Forest
rf_model = rf_pipeline.named_steps['classifier']
encoded_cat_cols = rf_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = list(numerical_features) + list(encoded_cat_cols)

importances = rf_model.feature_importances_
feature_imp_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("Top 5 Most Important Features in Predicting Churn:")
print(feature_imp_df.head(5))