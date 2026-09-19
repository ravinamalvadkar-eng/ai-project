📊 Telecom Customer Churn Prediction using Supervised ML
An end-to-end Machine Learning pipeline built with Python and scikit-learn to analyze customer subscription data, identify key churn drivers, and predict high-risk churn customers in the telecommunications industry.
📌 Table of Contents
Project Overview
Problem Statement
Dataset Architecture
Project Workflow
Model Evaluation & Results
Key Business Insights
Repository Structure
Getting Started
License
🎯 Project Overview
Acquiring new subscribers can cost up to 5 times more than retaining existing ones. Customer churn occurs when subscribers cancel their contracts or switch to competitors.
This project implements a complete supervised machine learning framework to:
Preprocess and impute raw secondary subscriber datasets.
Perform Exploratory Data Analysis (EDA) to uncover churn indicators.
Train and compare probabilistic binary classification models (Logistic Regression, Random Forest).
Extract actionable statistical insights to help telecom retention teams minimize customer acquisition overhead.
💡 Problem Statement
Given historical subscriber information, account attributes, and monthly usage metrics, build a binary classification model to predict whether a customer will Churn (1) or Stay (0).
📁 Dataset Architecture
Source: Telco Customer Churn Public Dataset (Kaggle / IBM)
Observations: 7,043 subscriber records
Features: 20 predictor variables + 1 target variable (Churn)
Feature Breakdown:


Feature Category
Features Included
Demographics
gender, SeniorCitizen, Partner, Dependents
Account Info
tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
Subscribed Services
PhoneService, MultipleLines, InternetService, OnlineSecurity, TechSupport, StreamingTV

⚙️ Project Workflow
┌─────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│  Data Ingestion │ ──► │ Preprocessing & Scaling│ ──► │  EDA & Visualizations│
└─────────────────┘     └──────────────────────┘     └──────────────────────┘
                                                                │
┌─────────────────┐     ┌──────────────────────┐                ▼
│ Insights & Docs │ ◄── │  Model Evaluation    │ ◄── ┌──────────────────────┐
└─────────────────┘     └──────────────────────┘     │ Supervised Modeling  │
                                                     └──────────────────────┘


Missing Value Imputation: Handled blank spaces in continuous fields using median imputation.
Feature Engineering & Encoding: Applied One-Hot Encoding (OneHotEncoder) to multi-class variables and standardized continuous features (Tenure, MonthlyCharges) via StandardScaler.
Data Splitting: Applied a 80/20 Stratified Split to ensure equal distribution of churn proportions in training and evaluation sets.
📈 Model Evaluation & Results
Two algorithms were evaluated using testing data ():
Algorithm
Accuracy
Precision
Recall
F1-Score
ROC-AUC
Logistic Regression
78.8%
61.2%
55.4%
58.1%
0.82
Random Forest Classifier 🏆
80.4%
65.2%
58.7%
61.8%
0.84

Best Performer: The Random Forest Classifier achieved the highest overall discriminatory power with an AUC score of 0.84.
🔑 Key Business Insights
Contract Duration: Customers on Month-to-Month contracts exhibit a 42.7% churn rate, compared to <3% for 2-year contracts.
Early Tenure Risk: Churn rates are significantly elevated within the first 6 months of subscription.
Payment Channels: Subscribers using Electronic Check as their payment method showed a higher propensity to churn than those on auto-debit or credit card billing.
📂 Repository Structure
├── data/                      # Dataset files
│   └── dataset.csv
├── src/                       # Source code scripts
│   └── main.py                # Full Python end-to-end ML pipeline
├── docs/                      # Generated HTML and PDF project reports
│   └── report.html
├── README.md                  # Comprehensive GitHub project documentation
├── requirements.txt           # Python dependency specifications
└── .gitignore                 # Files excluded from version control


