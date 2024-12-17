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
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

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
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data[['Quantity', 'UnitPrice']])
    labels = kmeans.labels_
    return labels

# Predictive analysis using Linear Regression
def predictive_analysis(data):
    X = data[['Quantity']]
    y = data['UnitPrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse, y_test, y_pred

# Decision Tree Regression
def decision_tree_regression(data):
    X = data[['Quantity']]
    y = data['UnitPrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse, y_test, y_pred

# Random Forest Regression
def random_forest_regression(data):
    X = data[['Quantity']]
    y = data['UnitPrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse, y_test, y_pred

# Visualize clusters
def visualize_clusters(data, labels):
    plt.figure(figsize=(8, 6))
    plt.scatter(
        data['Quantity'], 
        data['UnitPrice'], 
        c=labels, 
        cmap='viridis', 
        s=50
    )
    plt.title("Customer Segmentation")
    plt.xlabel("Quantity")
    plt.ylabel("UnitPrice")
    plt.colorbar(label='Cluster')
    plt.grid(True)
    st.pyplot(plt)

# Visualize predictions
def visualize_predictions(y_test, y_pred, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual')
    plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted')
    plt.title(f"{title}: Actual vs Predicted")
    plt.xlabel("Sample Index")
    plt.ylabel("UnitPrice")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

def main():
    st.title("Data Mining App")

    # Display the introduction
    st.header("Introduction")
    st.write("""
        This project demonstrates data mining techniques using Python libraries scikit-learn. 
        The goal is to perform customer segmentation and predictive analysis for an online retail store.
    """)

    st.header("Methods")
    st.write("""
        - **Data Preprocessing**: Handling missing values and normalization.
        - **Customer Segmentation**: KMeans clustering.
        - **Predictive Analysis**: Linear Regression.
        - **Classification**: Decision Tree and Random Forest.
    """)

    # File uploader
    uploaded_file = st.file_uploader("Upload your dataset (.xlsx file only)", type=["xlsx"])
    if uploaded_file is not None:
        data = load_data(uploaded_file)
    else:
        st.warning("Please upload a dataset file.")
        st.stop()

    # Preprocess data
    data = preprocess_data(data)

    # Customer segmentation
    st.header("Customer Segmentation")
    k = 3
    labels = customer_segmentation(data, k)
    st.write("Customer Segmentation Labels:", labels)
    visualize_clusters(data, labels)

    # Predictive analysis
    st.header("Predictive Analysis")
    mse_lr, y_test_lr, y_pred_lr = predictive_analysis(data)
    st.write("Linear Regression Mean Squared Error (MSE):", mse_lr)
    visualize_predictions(y_test_lr, y_pred_lr, "Linear Regression")

    # Decision Tree Regression
    st.header("Decision Tree Regression")
    mse_dt, y_test_dt, y_pred_dt = decision_tree_regression(data)
    st.write("Decision Tree Regression Mean Squared Error (MSE):", mse_dt)
    visualize_predictions(y_test_dt, y_pred_dt, "Decision Tree Regression")

    # Random Forest Regression
    st.header("Random Forest Regression")
    mse_rf, y_test_rf, y_pred_rf = random_forest_regression(data)
    st.write("Random Forest Regression Mean Squared Error (MSE):", mse_rf)
    visualize_predictions(y_test_rf, y_pred_rf, "Random Forest Regression")

if __name__ == "__main__":
    main()
