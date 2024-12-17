import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

# Load and preprocess data
@st.cache
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
        data['Sales'] = data['Quantity'] * data['UnitPrice']
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def preprocess_data(data):
    data.fillna({'CustomerID': 'Unknown', 'Description': 'Unknown'}, inplace=True)
    return data

# Visualization functions
def sales_by_country(data):
    sales_country = data.groupby('Country')['Sales'].sum().sort_values(ascending=False).reset_index()
    fig = px.bar(sales_country, x='Country', y='Sales', title='Total Sales by Country')
    st.plotly_chart(fig)

def top_selling_products(data):
    top_products = data.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10).reset_index()
    fig = px.bar(top_products, x='Description', y='Quantity', title='Top-Selling Products')
    st.plotly_chart(fig)

def sales_trend(data):
    sales_trend = data.resample('M', on='InvoiceDate')['Sales'].sum().reset_index()
    fig = px.line(sales_trend, x='InvoiceDate', y='Sales', title='Monthly Sales Trend')
    st.plotly_chart(fig)

# Main app
def main():
    st.title("Interactive Dashboard for Business Intelligence")

    # Sidebar for file upload and filters
    st.sidebar.header("Dashboard Settings")
    file_path = st.sidebar.file_uploader("Upload CSV Dataset", type=['csv'])
    st.sidebar.markdown("### Filters:")
    date_range = st.sidebar.date_input("Select Date Range", [])

    if file_path:
        # Load and preprocess data
        data = load_data(file_path)
        if data is not None:
            data = preprocess_data(data)

            # Apply date range filter if selected
            if len(date_range) == 2:
                data = data[(data['InvoiceDate'] >= pd.to_datetime(date_range[0])) &
                            (data['InvoiceDate'] <= pd.to_datetime(date_range[1]))]

            # Show data preview
            st.write("### Data Preview")
            st.dataframe(data.head())

            # Key metrics
            st.write("### Key Metrics")
            total_sales = data['Sales'].sum()
            unique_customers = data['CustomerID'].nunique()
            unique_products = data['StockCode'].nunique()
            st.metric("Total Sales", f"${total_sales:,.2f}")
            st.metric("Unique Customers", unique_customers)
            st.metric("Unique Products", unique_products)

            # Visualizations
            st.write("### Visualizations")
            st.write("#### Total Sales by Country")
            sales_by_country(data)

            st.write("#### Top-Selling Products")
            top_selling_products(data)

            st.write("#### Monthly Sales Trend")
            sales_trend(data)

if __name__ == "__main__":
    main()
