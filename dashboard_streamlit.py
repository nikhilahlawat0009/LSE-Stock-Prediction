#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os

# Set up the Streamlit page
st.set_page_config(page_title="Stock Forecasting Dashboard", layout="wide")
st.title("📈 Stock Forecasting Dashboard")

# Load processed data and models
data_folder = "processed_data"
models_folder = "models"
stock_files = [f for f in os.listdir(data_folder) if f.endswith("_processed.csv")]

# Dropdown to select a stock
stock_options = [f.replace("_processed.csv", "") for f in stock_files]
selected_stock = st.selectbox("Select a stock:", stock_options)

# Load stock data
df = pd.read_csv(os.path.join(data_folder, f"{selected_stock}_processed.csv"), index_col="Date", parse_dates=True)

# Plot historical stock prices
st.subheader("📊 Historical Stock Prices")
fig = px.line(df, x=df.index, y="Close", title=f"Closing Prices for {selected_stock}")
st.plotly_chart(fig, use_container_width=True)

# Load trained model
model_path = os.path.join(models_folder, f"{selected_stock}_rf_model.pkl")
if os.path.exists(model_path):
    model = joblib.load(model_path)
    
    # Prepare last available data for prediction
    feature_columns = ["Close", "SMA_50", "SMA_200", "EMA_20", "EMA_50", "RSI_14", "MACD", "Bollinger_Upper", "Bollinger_Lower"]
    latest_data = df[feature_columns].iloc[-1].to_frame().T  # Convert to DataFrame instead of array
    
    # Ensure feature names match training data
    latest_data.columns = feature_columns  
    
    predicted_price = model.predict(latest_data)[0]
    
    st.subheader("🔮 Next Day Predicted Price")
    st.metric(label=f"Predicted Closing Price for {selected_stock}", value=f"£{predicted_price:.2f}")
else:
    st.warning("No trained model found for this stock.")

st.subheader("📌 Technical Indicators")
st.dataframe(df[["SMA_50", "SMA_200", "EMA_20", "EMA_50", "RSI_14", "MACD", "Bollinger_Upper", "Bollinger_Lower"]].tail(10))


# In[ ]:




