import pandas as pd
import numpy as np
from scipy import stats
import os
import streamlit as st

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
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    
    # Remove duplicates
    data.drop_duplicates(inplace=True)
    
    return data

# Streamlit Application
def main():
    st.title("Online Retail Data Visualization")
    
    # Load dataset
    data = load_dataset(dataset_path)
    
    if data.empty:
        st.error("Dataset file not found or empty.")
        return
    
    cleaned_data = clean_transform_data(data)
    
    # Display Data
    st.subheader("Dataset")
    st.write(cleaned_data)

if __name__ == "__main__":
    main()