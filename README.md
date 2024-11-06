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
scaler = StandardScaler() <br>
numerical_features = data.select_dtypes(include=['float64', 'int64']).columns<br>
data[numerical_features] = scaler.fit_transform(data[numerical_features])<br>

## Step 4: Split the Data
X = data.drop(columns=['Churn' , 'customerID'])  <br>
y = data['Churn']<br>
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)<br>
## Step 5: Train the Model
IN this project i  used a Random Forest Classifier<br>
param_dist = {
    'n_estimators': [100, 200, 300, 400],<br>
    'max_depth': [10, 20, 30, None],<br>
    'min_samples_split': [2, 5, 10],<br>
    'min_samples_leaf': [1, 2, 4],<br>
    'max_features': ['auto', 'sqrt', 'log2']<br>
}

model = RandomForestClassifier(random_state=42, class_weight='balanced')<br>
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, cv=5, scoring='roc_auc', n_jobs=-1, random_state=42)<br>
random_search.fit(X_train, y_train)<br>

## Step 6: Making  Predictions and Evaluate this project 
y_train_pred = best_model.predict(X_train)<br>
y_test_pred = best_model.predict(X_test)<br>


print("Train Accuracy:", accuracy_score(y_train, y_train_pred) * 100)<br>
print("Test Accuracy:", accuracy_score(y_test, y_test_pred) * 100)<br>


print("Classification Report:\n", classification_report(y_test, y_test_pred))<br>
y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]<br>
print("AUC-ROC Score:", roc_auc_score(y_test, y_test_pred_proba))<br>
## Step 7: Visualize Results<br>
sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt='d', cmap='Blues')<br>
plt.xlabel('Predicted')<br>
plt.ylabel('Actual')<br>
plt.title('Confusion Matrix')<br>
plt.show()<br>
![pic3](https://github.com/user-attachments/assets/70d7f118-2a42-4496-9eed-b6a34fe1785f)<br>
![pic2](https://github.com/user-attachments/assets/79a1c369-3692-4806-b4be-fd34bcf70921)

## Step -7 Feature Importance (For Random Forest): 
![pic1](https://github.com/user-attachments/assets/27e4b9db-1034-43ba-bbec-57db4d83add1)<br>
## Step 8: Improve the Model : 
from sklearn.model_selection import GridSearchCV<br>

param_grid = {
    'n_estimators': [100, 200, 300],<br>
    'max_depth': [10, 20, 30, None],<br>
    'min_samples_split': [2, 5, 10],<br>
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='roc_auc')<br>
grid_search.fit(X_train, y_train)<br>
print("Best Parameters:", grid_search.best_params_)<br>



