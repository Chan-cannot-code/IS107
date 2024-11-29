import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data
def load_data(file_path):
    try:
        data = pd.read_excel(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Data preprocessing
def preprocess_data(data):
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
    
    non_numeric_cols = data.select_dtypes(exclude=['int64', 'float64']).columns
    data[non_numeric_cols] = data[non_numeric_cols].fillna('Unknown')
    
    return data

# Interactive data visualization
def visualize_data(data):
    fig = px.scatter(data, x='Quantity', y='UnitPrice', trendline='ols')
    st.plotly_chart(fig)

# Customer segmentation
def customer_segmentation(data, k):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data[['Quantity', 'UnitPrice']])
    labels = kmeans.labels_
    return labels

# Predictive analysis
def predictive_analysis(data, model):
    X = data[['Quantity']]
    y = data['UnitPrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

# Main app
def main():
    st.title("Business Intelligence Solution")

    # Sidebar
    st.sidebar.title("Settings")
    file_path = st.sidebar.file_uploader("Upload dataset", type=['xlsx'])
    k = st.sidebar.slider("Customer Segments", 2, 10)
    model_choice = st.sidebar.selectbox("Model", ["Linear Regression", "Decision Tree", "Random Forest"])

    if file_path:
        if st.sidebar.button('Load Data'):
            data = load_data(file_path)
            if data is not None:
                data = preprocess_data(data)
                
                # Visualization
                st.write("Data Visualization")
                visualize_data(data)

                # Customer segmentation
                st.write("Customer Segmentation")
                labels = customer_segmentation(data, k)
                st.write(labels)

                # Predictive analysis
                st.write("Predictive Analysis")
                if model_choice == "Linear Regression":
                    model = LinearRegression()
                elif model_choice == "Decision Tree":
                    model = DecisionTreeRegressor()
                else:
                    model = RandomForestRegressor()
                mse = predictive_analysis(data, model)
                st.write(f"MSE: {mse:.2f}")
            else:
                st.write("Error loading data.")

if __name__ == "__main__":
    main()