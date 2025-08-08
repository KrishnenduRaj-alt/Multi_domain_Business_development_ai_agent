import streamlit as st
import pandas as pd
from data_loader import load_medical_store_data # Import your data loader function
from datetime import datetime 
import random 
# --- Authentication (Placeholder) ---
# This is a simplified authentication for demonstration purposes only.
# DO NOT use this for production applications.
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.sidebar.empty() # Clear sidebar content for login screen
    st.title("🔒 Login to Multi Domain Business Development AI Agent")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "password": # Hardcoded credentials
            st.session_state.logged_in = True
            st.success("Logged in successfully!")
            st.rerun() # Rerun app to show dashboard
        else:
            st.error("Invalid username or password.")
    st.stop() # Stop further execution until logged in
# --- End Authentication ---
# Helper to suppress Prophet's verbose output during fitting
import os
import sys
from contextlib import contextmanager

@contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, "w") as fnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = fnull
        sys.stderr = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

st.set_page_config(
    page_title="Multi Domain Business Development AI Agent", # New browser tab title
    page_icon="🌐", # Changed to a globe for multi-domain, or you could use 🤖
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Title and Introduction ---
st.title("🤖 Business Assistant AI Agent 📊🧠📈") # New main title with multiple emojis
st.markdown(
    """
    Welcome to your intelligent business assistant! This dashboard provides real-time insights,
    sales trends, and inventory recommendations for your business across various domains.
    """
)

# --- Sidebar for Filters & Controls ---
st.sidebar.header("Filters & Controls")

# NEW: Domain Selection
st.sidebar.subheader("Business Domain")
selected_domain = st.sidebar.selectbox(
    "Select Business Type:",
    (
        "Medical Store",
        "Supermarket (Coming Soon)",
        "Hotel (Coming Soon)",
        "Electronics Shop (Coming Soon)",
        "Textile Shop (Coming Soon)",
        "Book Stall (Coming Soon)",
        "Automobile Rental (Coming Soon)",
        "Stationary Store (Coming Soon)",
        "Used Car Showroom (Coming Soon)",
        "Restaurant (Coming Soon)",
        "Fitness Center (Coming Soon)",
        "Salon/Spa (Coming Soon)",
        "Restobars/Pubs(coming soon)"
    )
)

# --- Load Data ---
# Use Streamlit's caching to load data only once and speed up app
@st.cache_data
def get_data(domain): # Modified to accept domain
    file_path = ""
    if domain == "Medical Store":
        file_path = "simulated_medical_store_data.csv"
    # Add more elif conditions here for other domains when you have their data
    # elif domain == "Supermarket (Coming Soon)":
    #     file_path = "simulated_supermarket_data.csv"
    # elif domain == "Hotel (Coming Soon)":
    #     file_path = "simulated_hotel_data.csv"
    # elif domain == "Electronics Shop (Coming Soon)":
    #     file_path = "simulated_electronics_data.csv"
    else:
        st.warning("Please select a valid business domain to load data.")
        return None # Return None if no valid domain selected

    return load_medical_store_data(file_path=file_path) # Pass the file_path to the loader

df = get_data(selected_domain) # Pass the selected domain to get_data

if df is None:
    st.error(f"Failed to load data for {selected_domain}. Please ensure the correct CSV file exists and the data loader is configured.")
    st.stop() # Stop the app if data loading fails




# --- Key Performance Indicators (KPIs) ---
st.markdown("---")
st.subheader("📊 Key Performance Indicators")

# Calculate KPIs
total_sales = df['total_sale'].sum()
total_quantity = df['quantity'].sum()
average_sale_value = df['total_sale'].mean()

# Display KPIs using Streamlit columns for a nice layout
col1, col2, col3 = st.columns(3) # Create 3 columns for KPIs

with col1:
    st.metric(label="Total Revenue (Last 2 Years)", value=f"£{total_sales:,.2f}")
with col2:
    st.metric(label="Total Items Sold (Last 2 Years)", value=f"{total_quantity:,.0f}")
with col3:
    st.metric(label="Average Sale Value", value=f"£{average_sale_value:,.2f}")

# --- Sales Trends ---
st.markdown("---")
st.subheader("📈 Sales Trends Over Time")

# Sidebar filter for date range
st.sidebar.subheader("Sales Trend Filters")
num_days = st.sidebar.slider(
    "Select number of recent days to display sales trend:",
    min_value=30, max_value=730, value=180 # Default to last 180 days (approx 6 months)
)

# Filter data based on selected number of days
end_date = df['date'].max()
start_date_filtered = end_date - pd.Timedelta(days=num_days)
filtered_df = df[(df['date'] >= start_date_filtered) & (df['date'] <= end_date)]

# Aggregate daily sales for the chart
daily_sales = filtered_df.groupby(pd.to_datetime(filtered_df['date'].dt.date))['total_sale'].sum().reset_index()
daily_sales.columns = ['Date', 'Daily Sales'] # Rename columns for clarity

if not daily_sales.empty:
    st.line_chart(daily_sales, x='Date', y='Daily Sales', use_container_width=True)
else:
    st.warning("No sales data available for the selected period.")

# --- Sales Forecasting ---
st.markdown("---")
st.subheader("🔮 Sales Forecasting")
st.write("Predicting future daily sales based on historical trends.")

# Prepare data for Prophet: needs 'ds' (datestamp) and 'y' (value) columns
# We'll use the daily_sales DataFrame we already created for the trend chart
# Ensure 'Date' is datetime and rename for Prophet
prophet_df = daily_sales.rename(columns={'Date': 'ds', 'Daily Sales': 'y'})

# Make sure 'ds' is truly datetime (it should be from daily_sales creation)
prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])

# Filter out any future dates if they somehow exist in the raw data (shouldn't for sales)
prophet_df = prophet_df[prophet_df['ds'] <= pd.to_datetime(df['date'].max().date())]

# Check if there's enough data for forecasting
if len(prophet_df) < 2: # Prophet needs at least 2 data points
    st.warning("Not enough data to generate a meaningful sales forecast. Please ensure you have at least 2 days of sales data.")
else:
    # Import Prophet only when needed (to avoid import errors if not installed)
    from prophet import Prophet
    from prophet.plot import plot_plotly
    import plotly.graph_objs as go

    # Create and fit the Prophet model
    # Suppress command line output from Prophet
    with suppress_stdout_stderr():
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False, # Daily seasonality is often less relevant for daily aggregates unless very specific patterns
            seasonality_mode='additive'
        )
        m.fit(prophet_df)

    # Create a DataFrame for future predictions (e.g., next 30 days)
    future = m.make_future_dataframe(periods=30) # Forecast for the next 30 days

    # Make predictions
    forecast = m.predict(future)

    # Display the forecast
    st.markdown("##### Daily Sales Forecast (Next 30 Days)")

    # Plotting with Plotly for interactivity
    fig = plot_plotly(m, forecast) # This generates a Plotly figure

    # Add a shaded region for the historical data range
    last_historical_date = prophet_df['ds'].max()
    fig.add_shape(
        type="rect",
        x0=prophet_df['ds'].min(),
        y0=0,
        x1=last_historical_date,
        y1=forecast['yhat'].max() * 1.1, # Extend slightly above max yhat for visibility
        fillcolor="LightSkyBlue",
        opacity=0.2,
        layer="below",
        line_width=0,
        name="Historical Data"
    )
    fig.add_annotation(
        x=last_historical_date - (last_historical_date - prophet_df['ds'].min()) / 2,
        y=forecast['yhat'].max() * 1.05,
        text="Historical Data",
        showarrow=False,
        font=dict(size=10, color="gray")
    )

    # Update layout for better readability
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Predicted Sales (£)",
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Adjust y-axis to start from 0 and ensure it covers predictions
    fig.update_yaxes(range=[0, forecast['yhat_upper'].max() * 1.1])

    st.plotly_chart(fig, use_container_width=True)

    # Display forecast components (optional, but good for understanding)
    with st.expander("View Forecast Components"):
        fig_components = m.plot_components(forecast)
        st.write(fig_components) # Streamlit can display matplotlib figures directly



# --- Top Selling Products and Categories ---
st.subheader("🏆 Top Selling Products & Categories")
# --- Anomaly Detection ---
st.markdown("---")
st.subheader("🚨 Anomaly Detection")
st.write("Identifying unusual patterns or outliers in daily sales data.")

# Import IsolationForest only when needed
from sklearn.ensemble import IsolationForest

# Prepare data for anomaly detection (using daily_sales from the trend section)
# We'll use 'Daily Sales' as the feature for anomaly detection
# Need to reshape for IsolationForest
X = daily_sales[['Daily Sales']]

if not X.empty:
    # Train Isolation Forest model
    # contamination: The proportion of outliers in the dataset. Adjust if you expect more/fewer anomalies.
    # random_state: For reproducibility of results.
    model = IsolationForest(contamination=0.01, random_state=42) # Assuming 1% of data are anomalies
    model.fit(X)

    # Predict anomalies (-1 for outliers, 1 for inliers)
    daily_sales['anomaly'] = model.predict(X)
    daily_sales['anomaly_score'] = model.decision_function(X) # Lower score = more anomalous

    # Filter out actual anomalies
    anomalies = daily_sales[daily_sales['anomaly'] == -1]

    st.markdown("##### Detected Sales Anomalies")

    if not anomalies.empty:
        # Display anomalies in a table
        st.warning(f"Found {len(anomalies)} potential sales anomalies:")
        st.dataframe(anomalies[['Date', 'Daily Sales', 'anomaly_score']].sort_values(by='anomaly_score'), use_container_width=True, hide_index=True)

        # Optional: Visualize anomalies on the sales trend chart
        st.markdown("##### Sales Trend with Anomalies Highlighted")

        # Create a Plotly figure to combine original trend and anomalies
        fig_anomaly = go.Figure()

        # Add normal sales data
        fig_anomaly.add_trace(go.Scatter(
            x=daily_sales['Date'],
            y=daily_sales['Daily Sales'],
            mode='lines',
            name='Daily Sales',
            line=dict(color='lightgray') # Make normal line subtle
        ))

        # Add anomalous points
        if not anomalies.empty:
            fig_anomaly.add_trace(go.Scatter(
                x=anomalies['Date'],
                y=anomalies['Daily Sales'],
                mode='markers',
                marker=dict(color='red', size=8, symbol='circle'),
                name='Anomaly',
                hoverinfo='text',
                text=[f"Date: {d.strftime('%Y-%m-%d')}<br>Sales: £{s:,.2f}<br>Anomaly Score: {score:.2f}" for d, s, score in zip(anomalies['Date'], anomalies['Daily Sales'], anomalies['anomaly_score'])]
            ))

        fig_anomaly.update_layout(
            xaxis_title="Date",
            yaxis_title="Daily Sales (£)",
            hovermode="x unified",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            title_text="Daily Sales with Anomalies (Red Dots)"
        )
        fig_anomaly.update_yaxes(range=[0, daily_sales['Daily Sales'].max() * 1.1])

        st.plotly_chart(fig_anomaly, use_container_width=True)

    else:
        st.info("No significant sales anomalies detected for the selected period.")
else:
    st.info("Not enough data to perform anomaly detection for the selected period.")

st.markdown("---") # Add a separator line
st.write("Day 4 dashboard elements are complete!") # A message to indicate progress
# Sidebar filter for number of top items
st.sidebar.subheader("Top Items Filters")
num_top_items = st.sidebar.slider(
    "Select number of top items to display:",
    min_value=5, max_value=20, value=10 # Default to top 10 items
)

# Use Streamlit columns to display products and categories side-by-side
col_prod, col_cat = st.columns(2)

with col_prod:
    st.markdown("##### Top Selling Products")
    # Group data by 'product_name', sum 'total_sale', and get the top N largest
    top_products = df.groupby('product_name')['total_sale'].sum().nlargest(num_top_items).reset_index()
    top_products.columns = ['Product', 'Total Sales'] # Rename columns for display
    if not top_products.empty:
        st.dataframe(top_products, use_container_width=True, hide_index=True)
    else:
        st.info("No top products to display.")

with col_cat:
    st.markdown("##### Top Selling Categories")
    # Group data by 'category', sum 'total_sale', and get the top N largest
    top_categories = df.groupby('category')['total_sale'].sum().nlargest(num_top_items).reset_index()
    top_categories.columns = ['Category', 'Total Sales'] # Rename columns for display
    if not top_categories.empty:
        st.dataframe(top_categories, use_container_width=True, hide_index=True)
    else:
        st.info("No top categories to display.")
# Line 472 (NEW CUSTOMER INSIGHTS CODE STARTS HERE)
# --- Customer Insights ---
st.markdown("---")
st.subheader("👥 Customer Insights")
st.write("Understanding your customer base.")

# Top Customers by Total Spending
top_spenders = df.groupby(['customer_id', 'customer_name'])['total_sale'].sum().nlargest(10).reset_index()
top_spenders.columns = ['Customer ID', 'Customer Name', 'Total Spent']

st.markdown("##### Top 10 Customers by Spending")
if not top_spenders.empty:
    st.dataframe(top_spenders, use_container_width=True, hide_index=True)
else:
    st.info("No customer spending data to display.")

st.markdown("---")

# Top Customers by Number of Visits/Transactions
top_visitors = df.groupby(['customer_id', 'customer_name'])['transaction_id'].nunique().nlargest(10).reset_index()
top_visitors.columns = ['Customer ID', 'Customer Name', 'Number of Visits']

st.markdown("##### Top 10 Customers by Number of Visits")
if not top_visitors.empty:
    st.dataframe(top_visitors, use_container_width=True, hide_index=True)
else:
    st.info("No customer visit data to display.")

# Line ~500 (Original Line 472, now shifted down)
st.markdown("---") # Add a separator line
# ... (rest of your Report Generation code and subsequent sections) ...
# --- Inventory Management & Restocking Suggestions ---
st.markdown("---")
st.subheader("📦 Inventory Management & Restocking")
st.write("Intelligent suggestions for inventory restocking and expiry tracking.")

# Simulate current stock levels (for demonstration, as our data is transactional history)
# In a real system, this would come from a live inventory database.
# We'll assume current stock is roughly 30 days of average sales for popular items,
# and less for others, with some randomness.

# Calculate average daily sales per product for recent period (e.g., last 90 days)
recent_sales_df = df[df['date'] >= (df['date'].max() - pd.Timedelta(days=90))]
avg_daily_sales_per_product = recent_sales_df.groupby('product_name')['quantity'].sum() / 90 # Average over 90 days

# Create a simulated current stock DataFrame
# For simplicity, let's assume all products exist in stock
# We'll create a dummy stock level for each product based on avg sales
# and then identify which ones are "low"

unique_products = df['product_name'].unique()
simulated_stock_data = []
for product in unique_products:
    # Assume initial stock is 1.5x average daily sales for 30 days, plus some random
    initial_stock = avg_daily_sales_per_product.get(product, 0) * 30 * random.uniform(0.8, 1.2)
    simulated_stock_data.append({
        'product_name': product,
        'current_stock': max(0, round(initial_stock + random.randint(-50, 50))), # Ensure non-negative
        'reorder_point': round(avg_daily_sales_per_product.get(product, 0) * 14) # Reorder when stock hits 14 days of sales
    })
simulated_stock_df = pd.DataFrame(simulated_stock_data)
simulated_stock_df = simulated_stock_df.set_index('product_name')

# Identify items for restocking
items_to_restock = simulated_stock_df[simulated_stock_df['current_stock'] < simulated_stock_df['reorder_point']].reset_index()
items_to_restock['suggested_order_qty'] = items_to_restock['reorder_point'] * 2 # Suggest ordering 2x reorder point
items_to_restock['suggested_order_qty'] = items_to_restock['suggested_order_qty'].apply(lambda x: max(10, round(x))) # Min order 10

st.markdown("##### 🛒 Items to Restock")
if not items_to_restock.empty:
    st.warning("The following items are low in stock and may need reordering:")
    st.dataframe(items_to_restock[['product_name', 'current_stock', 'reorder_point', 'suggested_order_qty']], use_container_width=True, hide_index=True)
else:
    st.info("All products appear to be sufficiently stocked based on current sales trends.")

# --- Expiring Products Tracking ---
st.markdown("##### ⏳ Products Nearing Expiry")
st.write("Monitor products that are nearing their expiration date to minimize waste.")

# Filter for products expiring within the next 90 days
today = pd.to_datetime(datetime.now().date())
expiring_soon_threshold = today + pd.Timedelta(days=90)

# Get unique products with their latest expiry date (or earliest if multiple batches)
# This is a simplification; real systems track expiry per batch.
# For now, we'll just check all transactions' expiry dates
expiring_products = df[df['expiry_date'] <= expiring_soon_threshold].copy()

if not expiring_products.empty:
    # Group by product and find the earliest expiry date among those expiring soon
    expiring_summary = expiring_products.groupby('product_name')['expiry_date'].min().reset_index()
    expiring_summary.columns = ['Product Name', 'Earliest Expiry Date']
    expiring_summary['Days Until Expiry'] = (expiring_summary['Earliest Expiry Date'] - today).dt.days

    # Sort by soonest expiry
    expiring_summary = expiring_summary.sort_values(by='Days Until Expiry').reset_index(drop=True)

    st.error("The following products are nearing their expiry date:")
    st.dataframe(expiring_summary[['Product Name', 'Earliest Expiry Date', 'Days Until Expiry']], use_container_width=True, hide_index=True)
else:
    st.info("No products are currently identified as nearing expiry within the next 90 days.")

# --- Report Generation ---
st.markdown("---")
st.subheader("📄 Generate Business Report")
st.write("Generate a comprehensive PDF report summarizing current insights.")

# Get data needed for the report (from the already calculated variables)
# KPIs
report_kpis = {
    "Total Revenue": f"£{total_sales:,.2f}",
    "Total Items Sold": f"{total_quantity:,.0f}",
    "Average Sale Value": f"£{average_sale_value:,.2f}"
}

# Top Products & Categories (ensure these DataFrames exist from previous sections)
# Recalculate them if they are not globally accessible or might be filtered
# For simplicity, we'll recalculate based on the full df for the report
all_time_top_products = df.groupby('product_name')['total_sale'].sum().nlargest(10).reset_index()
all_time_top_products.columns = ['Product', 'Total Sales']

all_time_top_categories = df.groupby('category')['total_sale'].sum().nlargest(10).reset_index()
all_time_top_categories.columns = ['Category', 'Total Sales']

# Forecast summary (use the 'forecast' DataFrame generated in the forecasting section)
# Ensure 'forecast' is available. If not, you might need to make it a global variable or pass it from get_data()
# For now, assuming 'forecast' is available from the forecasting block.
forecast_summary_for_report = pd.DataFrame()
if 'forecast' in locals(): # Check if forecast variable exists
    forecast_summary_for_report = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


# Import the generate_report function
from report_generator import generate_report

if st.button("Generate & Download PDF Report"):
    report_filename = f"Business_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    # Call the report generation function
    report_success = generate_report(
        kpis=report_kpis,
        top_products_df=all_time_top_products,
        top_categories_df=all_time_top_categories,
        forecast_summary_df=forecast_summary_for_report,
        output_filename=report_filename
    )

    if report_success:
        st.success(f"Report '{report_filename}' generated successfully!")
        # Provide a download button
        with open(report_filename, "rb") as file:
            btn = st.download_button(
                label="Download Report",
                data=file,
                file_name=report_filename,
                mime="application/pdf"
            )
    else:
        st.error("Failed to generate report. Check terminal for errors.")

# --- Conversational AI Assistant (Chatbot) ---
st.markdown("---")
st.subheader("💬 Conversational AI Assistant")
st.write("Ask questions about your business data in natural language.")


# --- Prepare Data Context for the AI (THIS IS THE MOVED BLOCK) ---
# These variables need to be defined *before* the chat_input loop
# Calculate total sales for the last 1 year specifically for context
# Line 499 (or similar)
# Calculate total sales for the last 1 year specifically for context
end_date_1y = df['date'].max()
start_date_1y = end_date_1y - pd.Timedelta(days=365)
total_sales_last_1_year = df[(df['date'] >= start_date_1y) & (df['date'] <= end_date_1y)]['total_sale'].sum()

# Calculate total sales for the last 6 months
start_date_6m = end_date_1y - pd.Timedelta(days=180) # Approximately 6 months
total_sales_last_6_months = df[(df['date'] >= start_date_6m) & (df['date'] <= end_date_1y)]['total_sale'].sum()


business_overview = f"""
Overall Business Metrics:
- Total Revenue (Last 2 Years): ${total_sales:,.2f}
- Total Revenue (Last 1 Year): ${total_sales_last_1_year:,.2f}
- Total Revenue (Last 6 Months): ${total_sales_last_6_months:,.2f}
- Total Items Sold (Last 2 Years): {total_quantity:,.0f}
- Average Sale Value (Overall): ${average_sale_value:,.2f}
"""

# NEW: Recent Daily Sales Context for Chatbot
# Use the 'daily_sales' DataFrame which is already calculated for the trend chart
recent_daily_sales_df = daily_sales.tail(30).copy() # Get last 30 days
if not recent_daily_sales_df.empty:
    # Format dates for better readability for the AI
    recent_daily_sales_df['Date'] = recent_daily_sales_df['Date'].dt.strftime('%Y-%m-%d')
    recent_daily_sales_context = f"""
Recent Daily Sales (Last 30 Days):
{recent_daily_sales_df.to_string(index=False)}
"""
else:
    recent_daily_sales_context = "No recent daily sales data available."


top_products_str = top_products.to_string(index=False) if not top_products.empty else "No top products data."
top_categories_str = top_categories.to_string(index=False) if not top_categories.empty else "No top categories data."
top_items_context = f"""
Top Selling Items:
{top_products_str}

Top Selling Categories:
{top_categories_str}
"""

forecast_context = ""
if 'forecast' in locals() and not forecast.empty:
    future_forecast = forecast[forecast['ds'] > df['date'].max()].head(7)
    if not future_forecast.empty:
        forecast_context = f"""
        Sales Forecast for next 7 days:
        {future_forecast[['ds', 'yhat']].to_string(index=False)}
        """
    else:
        forecast_context = "No future sales forecast available."
else:
    forecast_context = "Sales forecast data not available."

anomaly_context = ""
if 'anomalies' in locals() and not anomalies.empty:
    anomaly_context = f"""
    Detected Sales Anomalies:
    {anomalies[['Date', 'Daily Sales', 'anomaly_score']].to_string(index=False)}
    """
else:
    anomaly_context = "No significant sales anomalies detected recently."

restock_context = ""
if 'items_to_restock' in locals() and not items_to_restock.empty:
    restock_context = f"""
    Items identified for restocking:
    {items_to_restock[['product_name', 'current_stock', 'suggested_order_qty']].to_string(index=False)}
    """
else:
    restock_context = "No products currently identified as low in stock."

expiry_context = ""
if 'expiring_summary' in locals() and not expiring_summary.empty:
    expiry_context = f"""
    Products nearing expiry:
    {expiring_summary[['Product Name', 'Earliest Expiry Date', 'Days Until Expiry']].to_string(index=False)}
    """
else:
    expiry_context = "No products currently identified as nearing expiry."

inventory_context = f"""
Inventory Status:
{restock_context}
{expiry_context}
"""
# --- END OF MOVED CONTEXT BLOCK ---


# Initialize chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask me about your sales, inventory, or trends..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Handle simple greetings directly
    if prompt.lower() in ["hi", "hello", "hey", "hi there"]:
        response = "Hello! How can I assist you with your business data today?"
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
        st.stop() # Stop further execution for simple greetings

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            # Combine all context into a single prompt for the Hugging Face model
            full_context_prompt = f"""
            You are a helpful AI business assistant. Answer the user's question **strictly and concisely** using **only** the provided "Provided Business Data" below.
If the exact information is not explicitly present in the "Provided Business Data", state "The information is not available in the current dataset."
Do not invent or infer information.
            {business_overview}

            {recent_daily_sales_context}

            {top_items_context}

            {forecast_context}

            {anomaly_context}

            {inventory_context}

            Based on this data, please answer the user's question.
            If the question cannot be answered from the provided data, politely state that the information is not available in the current dataset.
            Keep your answers concise and directly relevant to the data.

            User's question: {prompt}
            """

            # --- Google Gemini 2.0 Flash API Inference ---
            import requests # Ensure requests is imported
            import json # Ensure json is imported

            try:
                # Construct the payload for the Gemini API
                payload = {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": full_context_prompt}]
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0.7, # Controls randomness (0.0 for deterministic, 1.0 for creative)
                        "maxOutputTokens": 500 # Max length of the AI's response
                    }
                }

                # API endpoint for Gemini 2.0 Flash
                # apiKey is left empty; Canvas environment will provide it at runtime
                api_key = "AIzaSyBrXmIAmaapJEIIneI9wlKHTzjYknzB-Ps" 
                api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

                # Make the POST request to the Gemini API
                response_raw = requests.post(api_url, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
                response_raw.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

                result = response_raw.json()

                if result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
                    response = result["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    response = "I'm sorry, I couldn't get a clear response from the AI model. The response structure was unexpected."

            except requests.exceptions.RequestException as e:
                response = f"An API request error occurred: {e}"
                st.error(response)
            except json.JSONDecodeError:
                response = "Failed to decode JSON response from AI model."
                st.error(response)
            except Exception as e:
                response = f"An unexpected error occurred: {e}"
                st.error(response)

            st.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown("---") # Final separator

# --- About Section ---
with st.expander("ℹ️ About Smart Pharma AI Agent"):
    st.write(
        """
        This application is an intelligent business development agent designed to provide
        actionable insights from transactional data. It's built to assist small and mid-sized
        businesses in making data-driven decisions.

        **Key Features:**
        - Real-time KPI monitoring
        - Sales trend analysis
        - Advanced sales forecasting
        - Anomaly detection for unusual patterns
        - Comprehensive PDF report generation
        - (Coming Soon: Conversational AI Interface, Inventory Restocking Suggestions, Domain-specific recommendations)

        **Developed by:** Krishnendu Raj (Student ID: 201809375)
        """
    )
    st.subheader("Contact")
    st.write("For support or inquiries, please contact: krishnendu7x463@gmail.com (Placeholder Email)")

st.markdown("---") # Another separator for the very bottom
st.write("Dashboard development for core features is complete!")


# --- Sidebar for Filters (will be expanded later) ---
st.sidebar.header("Filters & Controls")
st.sidebar.info("Filters and controls will appear here to customize your view.")
