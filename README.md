# AI-ML-Project

## Fraud Detection in Credit Card Transactions

## Introduction 
Fraudulent transactions pose a significant threat to both financial institutions and consumers. This project focuses on identifying fraudulent credit card transactions using anomaly detection techniques to enhance the security of online transactions.

## Tools Used
## Programming Language: Python
## Libraries:
1. Scikit-Learn for machine learning algorithms
2. XGBoost for classification
3. Pandas for data manipulation
4. Streamlit for web application development
5. Matplotlib and Seaborn for data visualization

## Steps Involved in Building the Project
1. Data Acquisition:
The Kaggle credit card dataset, containing 1,000,000 transactions, was utilized.

2. Preprocessing:
The dataset was loaded, and checks for null values and data types were performed.
Class distribution was analyzed to identify imbalances in fraudulent transactions.

3. Balancing the Dataset:
SMOTE (Synthetic Minority Over-sampling Technique) was applied to balance the training dataset.

4. Anomaly Detection:
Isolation Forest and Local Outlier Factor were implemented to detect anomalies in the dataset.

5. Model Training:
The XGBoost classifier was used for training the model on the balanced dataset.
Cross-validation was performed to evaluate the model's performance, achieving a mean F1 score of approximately 0.9988.

6. Model Evaluation:
A confusion matrix and classification report were generated to assess model accuracy, achieving a 100% accuracy.
The ROC curve was plotted to visualize the model's performance.

7. Dashboard Development:
An interactive dashboard was created using Streamlit to allow users to input transaction details for fraud prediction.

8. Deployment:
The Streamlit app was deployed on Streamlit Community Cloud, obtaining a permanent URL for public access.
