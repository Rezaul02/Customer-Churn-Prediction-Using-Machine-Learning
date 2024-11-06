## Customer-Churn-Prediction-Using-Machine-Learning
# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
# Step 2: Load and Inspect the Dataset
data = pd.read_csv('your_dataset.csv')
print(data.head())
print(data.info())
print(data.describe())
# Step 3: Data Preprocessing
In this part i checked the missing value have to Exist or not , if this misssing value have to present i removed it using dropna() function 
data.isnull().sum() 
data = data.dropna() 
# Encode Categorical Variables: Use label encoding or one-hot encoding for categorical features.
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Feature Scaling: Standardize or normalize numerical features for consistent scaling.

