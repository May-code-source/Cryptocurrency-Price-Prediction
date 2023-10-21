import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from joblib import load
from data_fetcher import Dataset
from datetime import datetime

# Load the saved model and scaler
regressor = load('joblib_model1.joblib')
scaler = load('joblib_scaler1.joblib')


def predict_smoothed(regressor, df, scaler, symbol_mapping, prediction_days=30):
    # Prepare the dataset for prediction
    df_copy = df.copy()
    df_copy = df_copy.reset_index()
    df_copy['symbol'] = df_copy['symbol'].map(symbol_mapping)
    df_scaled = scaler.transform(df_copy[['open', 'high', 'low', 'volume', 'symbol']])

    # Make the prediction
    predicted_values = regressor.predict(df_scaled)

    # Add the predicted values to the dataframe
    df_copy[['t1', 't7', 't30']] = predicted_values

    # Extract the prediction for the specified period
    if prediction_days == 1:
        predicted_value = df_copy['t1'].iloc[-1]
    elif prediction_days == 7:
        predicted_value = df_copy['t7'].iloc[-1]
    elif prediction_days == 30:
        predicted_value = df_copy['t30'].iloc[-1]

    return predicted_value, df_copy



# Load the symbols
symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOGEUSDT', 'DOTUSDT', 'XRPUSDT', 'LINKUSDT', 'LTCUSDT', 'BCHUSDT', 'UNIUSDT', 'MATICUSDT', 'SOLUSDT', 'VETUSDT', 'ETCUSDT', 'FILUSDT', 'THETAUSDT', 'XLMUSDT', 'TRXUSDT', 'EOSUSDT']

# Set Streamlit parameters
st.set_page_config(page_title='Crypto Forecast', layout='wide')

# title of the app
st.title("Crypto Forecast")

# Add a selectbox to the sidebar:
coin = st.sidebar.selectbox(
    'Select a cryptocurrency',
    symbols
)

# Add a selectbox for the prediction period:
period = st.sidebar.selectbox(
    'Select a prediction period',
    ['1 day', '7 days', '30 days']
)

# Get a fresh dataset
df = Dataset().get_data(ticker=coin, days=(datetime.now() - datetime(datetime.now().year - 1, 1, 1)).days, ts='1d')
df = pd.DataFrame(df)
df['symbol'] = coin


# Create a symbol_mapping dictionary
symbol_mapping = {symbol: i for i, symbol in enumerate(symbols)}

# Get prediction
prediction, df_pred = predict_smoothed(regressor, df, scaler, symbol_mapping, int(period.split()[0]))

# Display prediction
st.write(f"The predicted price for {coin} for the next {period} is ${prediction:.2f}")


# Create a Plotly figure
fig = go.Figure()

# Add the historical price line
fig.add_trace(go.Scatter(x=df_pred['time'], y=df_pred['close'], mode='lines', name='Historical Price'))

# Add the predicted price point
fig.add_trace(go.Scatter(x=[df_pred['time'].iloc[-1]], y=[prediction], mode='markers', name='Predicted Price', marker=dict(color='red', size=10)))

# Set the figure layout
fig.update_layout(title=coin, xaxis_title='Date', yaxis_title='Close Price USD ($)', showlegend=True)

# Display the figure
st.plotly_chart(fig)


def predict_smoothed(regressor, df, scaler, symbol_mapping, prediction_days=30):
    # Prepare the dataset for prediction
    df_copy = df.copy()
    df_copy = df_copy.reset_index()  # Since 'time' is used as index in your dataset
    df_copy['symbol'] = df_copy['symbol'].map(symbol_mapping)  # Convert the 'symbol' to integer
    df_scaled = scaler.transform(df_copy[['open', 'high', 'low', 'volume', 'symbol']])

    # Make the prediction
    predicted_values = regressor.predict(df_scaled)

    # Add the predicted values to the dataframe
    df_copy[['t1', 't7', 't30']] = predicted_values

    # Extract the prediction for the specified period
    if prediction_days == 1:
        predicted_value = df_copy['t1'].iloc[-1]
    elif prediction_days == 7:
        predicted_value = df_copy['t7'].iloc[-1]
    elif prediction_days == 30:
        predicted_value = df_copy['t30'].iloc[-1]

    return predicted_value, df_copy


def predict_smoothed(regressor, df, scaler, symbol_mapping, prediction_days=30):
    # Prepare the dataset for prediction
    df_copy = df.copy()
    df_copy = df_copy.reset_index()  # Since 'time' is used as index in your dataset
    df_copy['symbol'] = df_copy['symbol'].map(symbol_mapping)  # Convert the 'symbol' to integer
    df_scaled = scaler.transform(df_copy[['open', 'high', 'low', 'volume', 'symbol']])

    # Make the prediction
    predicted_values = regressor.predict(df_scaled)

    # Add the predicted values to the dataframe
    df_copy[['t1', 't7', 't30']] = predicted_values

    # Extract the prediction for the specified period
    if prediction_days == 1:
        predicted_value = df_copy['t1'].iloc[-1]
    elif prediction_days == 7:
        predicted_value = df_copy['t7'].iloc[-1]
    elif prediction_days == 30:
        predicted_value = df_copy['t30'].iloc[-1]

    return predicted_value, df_copy
