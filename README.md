# Titanic-ML-Classification-Model
# Titanic Survival Prediction – Machine Learning Model

This project focuses on building a Machine Learning classification model to predict the survival of passengers aboard the Titanic.  
The goal is to analyze the data, preprocess it, train a model, evaluate its performance, and visualize key insights.

## 📌 Project Overview
This project was created as part of **Task 9 – Data Analytics Internship (Skillytix)**.

The dataset contains passenger information such as:
- Age  
- Sex  
- Passenger Class  
- Fare  
- Embarkation Port  
- Number of Siblings/Spouses  
- Number of Parents/Children  

Using these features, a machine learning model predicts whether a passenger survived.

---

## 📂 Dataset
Dataset used: **Titanic - Machine Learning from Disaster**

Source (Kaggle):  
https://www.kaggle.com/c/titanic/data

Files used:
- `train.csv`
- `test.csv`

---

## 🛠️ Technologies Used
- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **Jupyter Notebook**

---

## 📊 Steps Performed

### **1. Data Loading**
Data was imported using Pandas and inspected with:
- `.head()`
- `.info()`
- `.describe()`
- `.isnull().sum()`

---

### **2. Data Cleaning & Preprocessing**
Performed:
- Dropped irrelevant columns → `['Cabin', 'Ticket', 'Name', 'PassengerId']`
- Filled missing values in:
  - **Age** → median  
  - **Embarked** → mode  
- Converted categorical data (`Sex`, `Embarked`) using **Label Encoding**

---

### **3. Feature Selection**
Separated data into:
- **Features (X)** → all independent columns  
- **Target (y)** → `Survived`

---

### **4. Train/Test Split**
Used an 80/20 split:
```python
train_test_split(X, y, test_size=0.2, random_state=42)

---
### **5.Model Used**

Logistic Regression, chosen because:

Simple

Fast

Works well for binary classification



---

### **6. Model Evaluation**

Evaluated using:

Accuracy Score

Classification Report

Confusion Matrix


Visualization of confusion matrix included.


---

📈 Model Performance

(Add your accuracy score here after training the model)

Example:

Accuracy: 0.81

Precision, Recall, F1-score included in classification report.



---

📊 Visualizations Included

Confusion Matrix Heatmap

Distribution Plots

Correlation Heatmap (optional)

Feature Exploration Charts



---

📁 Repository Structure

Titanic-ML-Classification-Model/
│
├── titanic_ml_model.ipynb     # Jupyter Notebook (main file)
├── README.md                   # Project documentation
├── data/                       # (optional) contains train.csv, test.csv
└── images/                     # (optional) graphs & plots


---

🚀 Conclusion

This project demonstrates:

End-to-end data preprocessing

Exploratory Data Analysis

Machine Learning model training

Evaluation using metrics and visualizations


It is a beginner-friendly yet powerful ML classification project suitable for portfolios and internships.


---

👤 Author

Kishlaynath Tiwari

GitHub: https://github.com/Kislaynath
LinkedIn: https://www.linkedin.com/in/kislay-tiwari-ba436838b/


---


