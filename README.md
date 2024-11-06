# Customer-Churn-Prediction-Using-Machine-Learning
## Step 1: Import Required Libraries
import pandas as pd <br>
import numpy as np<br>
from sklearn.model_selection import train_test_split<br>
from sklearn.preprocessing import StandardScaler, LabelEncoder<br>
from sklearn.ensemble import RandomForestClassifier<br>
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score<br>
import matplotlib.pyplot as plt<br>
import seaborn as sns<br>
## Step 2: Load and Inspect the Dataset
data = pd.read_csv('your_dataset.csv')<br>
print(data.head())<br>
print(data.info())<br>
print(data.describe())<br>
## Step 3: Data Preprocessing
In this part i checked the missing value have to Exist or not , if this misssing value have to present i removed it using dropna() function <br>
data.isnull().sum() <br>
data = data.dropna() <br>
## Encode Categorical Variables: Use label encoding or one-hot encoding for categorical features.
label_encoders = {}<br>
for column in data.select_dtypes(include=['object']).columns:<br>
    le = LabelEncoder()<br>
    data[column] = le.fit_transform(data[column])<br>
    label_encoders[column] = le<br>

## Feature Scaling: Standardize or normalize numerical features for consistent scaling.

