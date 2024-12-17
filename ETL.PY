import pandas as pd
import numpy as np
from scipy import stats
import os
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Dataset path
dataset_folder = 'LabelImg-20240325T152342Z-001'
dataset_file = 'Online Retail.xlsx'
dataset_path = os.path.join(os.getcwd(), dataset_folder, dataset_file)

# Load dataset
def load_dataset(dataset_path):
    try:
        data = pd.read_excel(dataset_path)
        return data
    except FileNotFoundError:
        print(f"File {dataset_path} not found.")
        return pd.DataFrame()  # Return empty DataFrame
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Clean and transform data
def clean_transform_data(data):
    if data.empty:
        return data

    # Handle missing values for numeric columns
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

    # Handle missing values for non-numeric columns
    non_numeric_columns = data.select_dtypes(include=['object']).columns
    data[non_numeric_columns] = data[non_numeric_columns].fillna('Unknown')

    # Detect outliers (z-score method)
    z_scores = np.abs(stats.zscore(data[numeric_columns]))

    # Align z_scores with data indices
    outlier_mask = (z_scores < 3).all(axis=1)
    data = data[outlier_mask]

    # Data type conversions
    if 'InvoiceDate' in data.columns:
        data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], errors='coerce')

    # Remove duplicates
    data.drop_duplicates(inplace=True)

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
    plt.scatter(range(len(y_test)), y_test, label="Actual", color='blue', alpha=0.6)
    plt.scatter(range(len(y_pred)), y_pred, label="Predicted", color='red', alpha=0.6)
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("UnitPrice")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

# Streamlit Application
def main():
    st.title("Data Mining App")

    # Load dataset
    data = load_dataset(dataset_path)

    if data.empty:
        st.error("Dataset file not found or empty.")
        return

    # Clean and transform data
    cleaned_data = clean_transform_data(data)

    # Display data
    st.subheader("Dataset")
    st.write(cleaned_data)

    # Customer segmentation
    k = 3
    if 'Quantity' in cleaned_data.columns and 'UnitPrice' in cleaned_data.columns:
        labels = customer_segmentation(cleaned_data, k)
        st.subheader("Customer Segmentation")
        st.write("Customer Segmentation Labels:", labels)
        visualize_clusters(cleaned_data, labels)
    else:
        st.warning("Columns 'Quantity' and 'UnitPrice' are required for customer segmentation.")

    # Predictive analysis - Linear Regression
    if 'Quantity' in cleaned_data.columns and 'UnitPrice' in cleaned_data.columns:
        mse_lr, y_test_lr, y_pred_lr = predictive_analysis(cleaned_data)
        st.subheader("Linear Regression")
        st.write("Mean Squared Error (MSE):", mse_lr)
        visualize_predictions(y_test_lr, y_pred_lr, "Linear Regression")

        # Decision Tree Regression
        mse_dt, y_test_dt, y_pred_dt = decision_tree_regression(cleaned_data)
        st.subheader("Decision Tree Regression")
        st.write("Mean Squared Error (MSE):", mse_dt)
        visualize_predictions(y_test_dt, y_pred_dt, "Decision Tree Regression")

        # Random Forest Regression
        mse_rf, y_test_rf, y_pred_rf = random_forest_regression(cleaned_data)
        st.subheader("Random Forest Regression")
        st.write("Mean Squared Error (MSE):", mse_rf)
        visualize_predictions(y_test_rf, y_pred_rf, "Random Forest Regression")
    else:
        st.warning("Columns 'Quantity' and 'UnitPrice' are required for predictive analysis.")

if __name__ == "__main__":
    main()
# Clean and transform data
cleaned_data = clean_transform_data(data)

# Save the cleaned dataset
if not cleaned_data.empty:
    cleaned_file_path = os.path.join(os.getcwd(), dataset_folder, 'cleaned_dataset.csv')
    cleaned_data.to_csv(cleaned_file_path, index=False)
    print(f"Cleaned dataset saved at: {cleaned_file_path}")
else:
    print("Cleaned dataset is empty. Nothing to save.")
