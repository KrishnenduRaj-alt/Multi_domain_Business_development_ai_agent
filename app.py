import streamlit as st
import pandas as pd
from data_loader import load_medical_store_data # Import your data loader function

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
# --- Page Configuration (Optional but good for aesthetics) ---
st.set_page_config(
    page_title="Smart Pharma AI Agent",
    page_icon="💊", # A pill emoji as icon
    layout="wide", # Use a wide layout for the dashboard
    initial_sidebar_state="expanded" # Keep sidebar expanded by default
)

# --- Load Data ---
# Use Streamlit's caching to load data only once and speed up app
@st.cache_data # Use st.cache_data for data loading functions
def get_data():
    return load_medical_store_data()

df = get_data()

if df is None:
    st.error("Failed to load data. Please check 'simulated_medical_store_data.csv' and 'data_loader.py'.")
    st.stop() # Stop the app if data loading fails

# --- Title and Introduction ---
st.title("💊 Smart Pharma AI Agent")
st.markdown(
    """
    Welcome to your intelligent business assistant! This dashboard provides real-time insights,
    sales trends, and inventory recommendations for your medical store.
    """
)

# You can display the raw data for now to confirm it's loaded
# st.subheader("Raw Data Preview")
# st.dataframe(df.head()) # Display first few rows of the DataFrame

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

st.markdown("---") # Add a separator line
st.write("Day 3 dashboard elements are complete!") # A message to indicate progress


# --- Sidebar for Filters (will be expanded later) ---
st.sidebar.header("Filters & Controls")
st.sidebar.info("Filters and controls will appear here to customize your view.")
