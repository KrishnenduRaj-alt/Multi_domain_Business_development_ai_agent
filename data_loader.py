import pandas as pd

def load_medical_store_data(file_path="simulated_medical_store_data.csv"):
    """
    Loads medical store data from a CSV file into a pandas DataFrame.
    Performs basic cleaning and type conversion, specifically for medical data.
    """
    try:
        df = pd.read_csv(file_path)

        # Convert date columns to datetime objects
        df['date'] = pd.to_datetime(df['date'])
        df['expiry_date'] = pd.to_datetime(df['expiry_date'])

        # Ensure numeric types for calculations
        df['price'] = pd.to_numeric(df['price'])
        df['quantity'] = pd.to_numeric(df['quantity'])
        df['total_sale'] = pd.to_numeric(df['total_sale'])

        # Convert boolean-like column if needed (though our simulator outputs actual booleans)
        # df['prescription_needed'] = df['prescription_needed'].astype(bool)

        print(f"Successfully loaded {len(df)} records from {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please ensure it exists in the project root.")
        return None
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return None

if __name__ == "__main__":
    # This block runs only when data_loader.py is executed directly
    # It's good for testing the data loading functionality
    df_loaded = load_medical_store_data()
    if df_loaded is not None:
        print("\nFirst 5 rows of loaded data:")
        print(df_loaded.head())
        print("\nData Info after loading and initial processing:")
        print(df_loaded.info())