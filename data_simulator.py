import pandas as pd
from faker import Faker
import random
from datetime import datetime, timedelta

def generate_medical_store_data(num_records=50000, num_unique_customers=5000):
    fake = Faker('en_GB') # Using en_GB for UK context

    # --- Create a pool of unique customers first ---
    customer_pool = []
    for i in range(num_unique_customers):
        customer_pool.append({
            "customer_id": f"CUST{i:05d}", # Unique customer ID (e.g., CUST00001)
            "customer_name": fake.name(),
            "customer_email": fake.email(),
            "customer_location": fake.city() # Added for potential future use
        })
    # ------------------------------------------------

    products = {
        "Paracetamol 500mg": {"category": "Pain Relief", "base_price": 2.50, "prescription_needed": False},
        "Ibuprofen 200mg": {"category": "Pain Relief", "base_price": 3.20, "prescription_needed": False},
        "Amoxicillin 250mg": {"category": "Antibiotics", "base_price": 15.00, "prescription_needed": True},
        "Ventolin Inhaler": {"category": "Respiratory", "base_price": 12.00, "prescription_needed": True},
        "Multivitamin Tablets": {"category": "Supplements", "base_price": 8.50, "prescription_needed": False},
        "Bandages (Assorted)": {"category": "First Aid", "base_price": 4.00, "prescription_needed": False},
        "Antiseptic Cream": {"category": "First Aid", "base_price": 6.00, "prescription_needed": False},
        "Blood Pressure Monitor": {"category": "Medical Devices", "base_price": 35.00, "prescription_needed": False},
        "Thermometer (Digital)": {"category": "Medical Devices", "base_price": 9.00, "prescription_needed": False},
        "Cough Syrup (Adult)": {"category": "Cough & Cold", "base_price": 7.00, "prescription_needed": False},
        "Sudafed Decongestant": {"category": "Cough & Cold", "base_price": 5.50, "prescription_needed": False},
        "Laxatives (Gentle)": {"category": "Digestive Health", "base_price": 4.50, "prescription_needed": False},
        "Probiotic Capsules": {"category": "Digestive Health", "base_price": 10.00, "prescription_needed": False},
        "Insulin Syringes": {"category": "Diabetes Care", "base_price": 18.00, "prescription_needed": True},
        "Glucose Test Strips": {"category": "Diabetes Care", "base_price": 25.00, "prescription_needed": True},
        "Nurofen for Children": {"category": "Paediatric", "base_price": 4.80, "prescription_needed": False},
        "Zinc Supplements": {"category": "Supplements", "base_price": 7.00, "prescription_needed": False}
    }
    product_names = list(products.keys())

    data = []
    start_date = datetime.now() - timedelta(days=365 * 2) # Data for the last 2 years

    for i in range(num_records):
        transaction_date = start_date + timedelta(days=random.randint(0, 365 * 2 - 1),
                                                hours=random.randint(0, 23),
                                                minutes=random.randint(0, 59))
        product_name = random.choice(product_names)
        product_info = products[product_name]
        category = product_info["category"]
        base_price = product_info["base_price"]
        prescription_needed = product_info["prescription_needed"]

        price = round(base_price * random.uniform(0.9, 1.1), 2)
        quantity = random.randint(1, 3) # Typically smaller quantities for medical items

        # Generate an expiry date for medical products (usually in the future)
        if "Medical Devices" in category or "First Aid" in category:
            expiry_date = transaction_date + timedelta(days=random.randint(365 * 3, 365 * 8)) # Longer expiry for devices/first aid
        else: # For medicines/supplements
            if random.random() < 0.7:
                expiry_date = transaction_date + timedelta(days=random.randint(365, 365 * 2)) # 1-2 years
            else:
                expiry_date = transaction_date + timedelta(days=random.randint(365 * 3, 365 * 5)) # 3-5 years

        transaction_id = fake.uuid4()

        # --- Select a customer from the pool ---
        selected_customer = random.choice(customer_pool)
        customer_id = selected_customer["customer_id"]
        customer_name = selected_customer["customer_name"]
        customer_email = selected_customer["customer_email"]
        # ---------------------------------------

        data.append({
            "transaction_id": transaction_id,
            "date": transaction_date,
            "product_name": product_name,
            "category": category,
            "price": price,
            "quantity": quantity,
            "total_sale": round(price * quantity, 2),
            "prescription_needed": prescription_needed,
            "expiry_date": expiry_date,
            "customer_id": customer_id, # New/Updated field
            "customer_name": customer_name,
            "customer_email": customer_email,
            "store_location": random.choice(["Central London", "Manchester North", "Birmingham East", "Glasgow West"]),
            "payment_method": random.choice(["Credit Card", "Cash", "Online Payment", "Insurance"])
        })

    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    print("Generating simulated medical store data with repeat customers...")
    # Generate 50,000 records using 5,000 unique customers
    simulated_df = generate_medical_store_data(num_records=50000, num_unique_customers=5000)
    output_file = "simulated_medical_store_data.csv"
    simulated_df.to_csv(output_file, index=False)
    print(f"Simulated data saved to {output_file} with {len(simulated_df)} records.")
    print("Number of unique customers generated:", simulated_df['customer_id'].nunique())
    print("\nFirst 5 rows of generated data:")
    print(simulated_df.head())
    print("\nData Info:")
    print(simulated_df.info())