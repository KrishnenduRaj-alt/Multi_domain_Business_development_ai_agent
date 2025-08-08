import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Load your dataset
df = pd.read_csv("simulated_medical_store_data_updated.csv")

# 2. Turn each row into a "document" string for the chatbot
docs = []
for _, row in df.iterrows():
    text = (
        f"Transaction ID: {row['transaction_id']}, "
        f"Date: {row['date']}, Month: {row['month']}, Year: {row['year']}, "
        f"Product: {row['product_name']}, Category: {row['category']}, "
        f"Price: {row['price']}, Quantity: {row['quantity']}, Total Sale: {row['total_sale']}, "
        f"Profit Margin: {row['profit_margin']}, Total Profit: {row['total_profit']}, "
        f"Prescription Needed: {row['prescription_needed']}, "
        f"Expiry Date: {row['expiry_date']}, Supplier: {row['supplier']}, "
        f"Tags: {row['tags']}, Stock Remaining: {row['stock_remaining']}, "
        f"Customer ID: {row['customer_id']}, Customer Name: {row['customer_name']}, "
        f"Store Location: {row['store_location']}, Payment Method: {row['payment_method']}"
    )
    docs.append(text)

# 3. Split into chunks so the chatbot doesn't get overwhelmed
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
split_docs = text_splitter.create_documents(docs)

# 4. Create embeddings for each chunk (turn text into numbers)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 5. Store in FAISS vector database
vectorstore = FAISS.from_documents(split_docs, embeddings)

# 6. Save the database for later use
vectorstore.save_local("medical_store_faiss")
print("✅ Data processed and saved to FAISS vector database.")
