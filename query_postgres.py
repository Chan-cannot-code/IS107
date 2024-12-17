import pandas as pd
import psycopg2
import matplotlib.pyplot as plt

# Database connection parameters
db_params = {
    'dbname': 'sakamoto', 
    'user': 'postgres',
    'password': 'admin',  
    'host': 'localhost',
    'port': 5433          
}

try:
    # Connect to the PostgreSQL database
    conn = psycopg2.connect(**db_params)
    print("Database connection successful!")

    # Example SQL query: Total Sales by Country
    query = """
    SELECT cd.country, SUM(sf.sales_amount) AS total_sales
    FROM sales_fact sf
    JOIN customer_dim cd ON sf.customerid = cd.customerid
    GROUP BY cd.country
    ORDER BY total_sales DESC;
    """

    # Execute the query and load results into a pandas DataFrame
    df = pd.read_sql(query, conn)
    print("Query Result:")
    print(df)

    # Visualization: Total Sales by Country
    plt.figure(figsize=(10, 6))
    plt.bar(df['country'], df['total_sales'], color='skyblue')
    plt.title("Total Sales by Country")
    plt.xlabel("Country")
    plt.ylabel("Sales Amount")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Close the database connection
    if 'conn' in locals() and conn:
        conn.close()
        print("Database connection closed.")
