# Predicting-Customer-Churn

This project will implement all aspects learned in Data601.

# Overview

This project uses the Telco Customer Churn dataset to predict whether a customer will churn or not, that is, if a customer will exit or not. 
Customer retention is a critical part of keeping a business profitable. 
This study helps to predict which customers are likely churn or switch to a competitor. 
This is useful because the telecommunication business can improve its services to reduce the number of customers that churn.

The 5-fold cross-validation and the GridSearch CV will be used to access which model produces the best accuracy score.  

## Goals
- The goal is to train a classfication model that has an accuracy score of more than 0.73. 

The metrics used will be the accuracy score and the confusion matrix.

We will use 5-fold Cross-Validation to evaluate which model has the best accuracy score. 
Then use the GridSearch CV to fine tune the models. 


## Motivation and Background
This project is interesting because companies spending alot of money on marketing strategies in order to get prospective customers. 

Therefore, it is critical that once they get the customers they are able to retain them. 

According a study done, business that retain their customers were more profitable than those that did not.

This project is useful to businesses because it helps them see which customers are likely to churn out and improve on services that can reduce the number of customers who churn out.

This study will be useful for anyone wanting to learn how to make predictions using categorical data.

It is also useful for anyone wanting to learn how to select a model using K-fold Cross-Validation and GridSearch CV.

Other people have done work on this data set, for example in the [Predict Customer Churn](https://datascienceplus.com/predict-customer-churn-logistic-regression-decision-tree-and-random-forest/) the data was modeled using Logistic Regression, Decision Tree and Random Forest using R. In the [Telco Customer Churn Prediction](https://towardsdatascience.com/telco-customer-churn-prediction-72f5cbfb8964) the Logistic Regression model is used to analyze the dataset. 


## Table of Contents
Notebook:

[Final Technical Report](https://github.com/cko-976/Predicting-Customer-Churn/blob/main/Notebooks/Technical%20Report.ipynb). This notebook is submitted as part of the Data601 project. It includes the summary of the work done on the project. 

Please note the [EDA](https://github.com/cko-976/Predicting-Customer-Churn/blob/main/Notebooks/EDA.ipynb) notebook is available. 

## Data
[Getting Data.ipynb](https://github.com/cko-976/Predicting-Customer-Churn/blob/main/Notebooks/Getting%20Data.ipynb). Dataset from Kaggle.

This is the Telco Customer Churn dataset. This data is from Kaggle website. 
According to the information in the metadata it was created and updated 2018-02-23.
The data is a .csv file. It has 21 attributes and 7043 rows. 

If you would like to access this data please go to the [Kaggle website](https://www.kaggle.com/blastchar/telco-customer-churn).
You will need to sign in to access the data.

The following description is included with the dataset:
>
>**Context**
>
> "Predict behavior to retain customers. You can analyze all relevant customer data and develop focused customer retention programs." [IBM Sample Data Sets]
>
>**Content**
>Each row represents a customer, each column contains customer’s attributes described on the column Metadata.
> The data set includes information about:
>> - Customers who left within the last month – the column is called Churn
>> - Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
>> - Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
>> - Demographic info about customers – gender, age range, and if they have partners and dependents
>
>**Inspiration**
>To explore this type of models and learn more about the subject. 

## Software Requirements

  **Libraries:** Numpy, Pandas, Matplotlib, Sklearn, Seaborn
 
- from  Sklearn - LogisticRegression, ColumnTransformer, OneHotEncoder, DecisionTreeClassifier, RandomForest, LabelEncoder, 
CrossValidate, train_test_split, Confusion Matrix, Accuracy_score, GridSearch CV, K-nearest neighbors,Support Vector Machine.

