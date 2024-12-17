import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import tempfile


# Function to load dataset
def load_data(file_path):
    return pd.read_excel(file_path)


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


# Customer segmentation using KMeans
def customer_segmentation(data, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data[['Quantity', 'UnitPrice']])
    labels = kmeans.labels_
    return labels


# Visualize customer segmentation
def visualize_clusters(data, labels):
    plt.figure(figsize=(8, 6))
    plt.scatter(data['Quantity'], data['UnitPrice'], c=labels, cmap='viridis', s=50)
    plt.title("Customer Segmentation")
    plt.xlabel("Quantity")
    plt.ylabel("UnitPrice")
    plt.colorbar(label='Cluster')
    plt.grid(True)
    st.pyplot(plt)


# Predictive Analysis: Linear Regression
def predictive_analysis(data):
    X = data[['Quantity']]
    y = data['UnitPrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse, y_test, y_pred


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


# Main app layout
def main():
    st.title("Business Intelligence Web Application")

    # Sidebar navigation
    menu = ["Upload Dataset", "Data Visualization", "Data Mining"]
    choice = st.sidebar.selectbox("Navigation", menu)

    if choice == "Upload Dataset":
        st.header("Upload Dataset")
        uploaded_file = st.file_uploader("Upload your dataset (.xlsx)", type=["xlsx"])
        
        if uploaded_file:
            # Save uploaded file to temporary location
            if "dataset" not in st.session_state:
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(uploaded_file.getbuffer())
                    st.session_state.dataset = temp_file.name
            
            # Load and preview the data
            data = load_data(st.session_state.dataset)
            st.write("Dataset uploaded successfully!")
            st.dataframe(data.head())

    elif choice == "Data Visualization":
        st.header("Data Visualization")
        if "dataset" in st.session_state:
            data = load_data(st.session_state.dataset)
            data = preprocess_data(data)

            # Perform customer segmentation
            st.subheader("Customer Segmentation")
            k = st.slider("Number of Clusters", 2, 10, 3)
            labels = customer_segmentation(data, k)
            st.write("Cluster Labels:", labels)
            visualize_clusters(data, labels)
        else:
            st.error("Please upload a dataset first in the Upload Dataset section.")

    elif choice == "Data Mining":
        st.header("Data Mining")
        if "dataset" in st.session_state:
            data = load_data(st.session_state.dataset)
            data = preprocess_data(data)

            # Predictive Analysis
            st.subheader("Predictive Analysis: Linear Regression")
            mse, y_test, y_pred = predictive_analysis(data)
            st.write("Mean Squared Error (MSE):", mse)
            visualize_predictions(y_test, y_pred, "Linear Regression")
        else:
            st.error("Please upload a dataset first in the Upload Dataset section.")


if __name__ == "__main__":
    main()
