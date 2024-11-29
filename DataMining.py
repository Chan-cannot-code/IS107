# Import libraries
import pandas as pd
import numpy as np
import streamlit as st
import os
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error

# Load dataset
def load_data(file_path):
    data = pd.read_excel(file_path)
    return data

# Data preprocessing
def preprocess_data(data):
    # Handle missing values for numeric columns
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
    
    # Handle missing values for non-numeric columns
    non_numeric_cols = data.select_dtypes(exclude=['int64', 'float64']).columns
    data[non_numeric_cols] = data[non_numeric_cols].fillna('Unknown')
    
    # Normalize data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    data[['Quantity', 'UnitPrice']] = scaler.fit_transform(data[['Quantity', 'UnitPrice']])
    
    return data

# Customer segmentation using KMeans clustering
def customer_segmentation(data, k):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data[['Quantity', 'UnitPrice']])
    labels = kmeans.labels_
    return labels

# Predictive analysis using Linear Regression
def predictive_analysis(data):
    X = data[['Quantity']]
    y = data['UnitPrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

# Decision Tree Regression
def decision_tree_regression(data):
    X = data[['Quantity']]
    y = data['UnitPrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    dt = DecisionTreeRegressor()
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

# Random Forest Regression
def random_forest_regression(data):
    X = data[['Quantity']]
    y = data['UnitPrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

# Streamlit app
def main():
    st.title("Data Mining App")
    
    # Load dataset
    dataset_folder = 'LabelImg-20240325T152342Z-001'
    dataset_file = 'Online Retail.xlsx'
    file_path = os.path.join(os.getcwd(), dataset_folder, dataset_file)
    data = load_data(file_path)
    
    # Preprocess data
    data = preprocess_data(data)
    
    # Customer segmentation
    k = 3
    labels = customer_segmentation(data, k)
    st.write("Customer Segmentation Labels:", labels)
    
    # Predictive analysis
    mse_lr = predictive_analysis(data)
    st.write("Linear Regression Mean Squared Error (MSE):", mse_lr)
    
    # Decision Tree Regression
    mse_dt = decision_tree_regression(data)
    st.write("Decision Tree Regression Mean Squared Error (MSE):", mse_dt)
    
    # Random Forest Regression
    mse_rf = random_forest_regression(data)
    st.write("Random Forest Regression Mean Squared Error (MSE):", mse_rf)

if __name__ == "__main__":
    main()