# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import mplfinance as mpf

# Title of the app
st.title('Stock Trend Prediction')

# User input for stock ticker, start date, and end date
user_input = st.text_input('Enter Stock Ticker', 'AAPL')
start_date = st.date_input('Start Date', value=pd.to_datetime('2010-01-01'))
end_date = st.date_input('End Date', value=pd.to_datetime('2024-04-01'))

# Download data
df = yf.download(user_input, start=start_date, end=end_date)

# Check if the DataFrame contains sufficient data
if df.empty or len(df) < 100:
    st.error("Insufficient data available for the specified date range. Please adjust the start and end dates.")
else:
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])

    # Calculate moving averages
    ma10 = df['Close'].rolling(window=10).mean()
    ma20 = df['Close'].rolling(window=20).mean()
    ma50 = df['Close'].rolling(window=50).mean()
    ma100 = df['Close'].rolling(window=100).mean()
    ma200 = df['Close'].rolling(window=200).mean()

    # Plot closing price and moving averages
    st.subheader('Closing Price VS Time Chart with Moving Averages')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Close'], 'b', label='Closing Price')
    ax.plot(ma10, 'purple', label='10 MA')
    ax.plot(ma20, 'orange', label='20 MA')
    ax.plot(ma50, 'green', label='50 MA')
    ax.plot(ma100, 'red', label='100 MA')
    ax.plot(ma200, 'yellow', label='200 MA')
    ax.legend()
    st.pyplot(fig)

    # Candlestick chart with moving averages and volume
    st.subheader('Candlestick Chart with Moving Averages and Volume')
    fig, axlist = mpf.plot(df.set_index('Date'), type='candle', mav=(10, 20, 50), volume=True, style='charles', returnfig=True)
    st.pyplot(fig)

    # Data split into training and testing sets
    train_size = int(len(df) * 0.70)
    data_training = pd.DataFrame(df['Close'][:train_size])
    data_testing = pd.DataFrame(df['Close'][train_size:])

    # Data scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_array = scaler.fit_transform(data_training)

    # Load pre-trained LSTM model
    model = load_model('keras.h5')

    # Prepare testing data
    past_100_days = data_training.tail(100)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
    input_data = scaler.transform(final_df)

    x_test = []
    y_test = []

    # Ensure we have sufficient data for predictions
    for i in range(100, len(input_data)):
        x_test.append(input_data[i - 100: i])
        y_test.append(input_data[i, 0])

    if x_test:
        x_test, y_test = np.array(x_test), np.array(y_test)

        # Model predictions
        y_predicted = model.predict(x_test)

        # Rescale the predictions
        scale_factor = 1 / scaler.scale_[0]
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor

        # Plot predictions vs original prices
        st.subheader('Predictions vs. Original Prices')
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(y_test, 'b', label='Original Price')
        ax2.plot(y_predicted, 'r', label='Predicted Price')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Price')
        ax2.legend()
        st.pyplot(fig2)

    else:
        st.error("Insufficient data for predictions. Please choose a different date range.")
        
    # Daily prediction function
    def daily_prediction():
        # Use the last 100 days of the dataset as input for prediction
        last_100_days = df.tail(100)['Close']
        last_100_days_scaled = scaler.transform(last_100_days.values.reshape(-1, 1))

        # Prepare input data
        x_input = np.array([last_100_days_scaled]).reshape((1, 100, 1))

        # Predict the next day's closing price
        predicted_price = model.predict(x_input)
        predicted_price = predicted_price * scale_factor  # Rescale the prediction

        # Display the daily prediction
        st.subheader('Daily Prediction')
        st.write(f'Predicted closing price for tomorrow: ${predicted_price[0][0]:.2f}')

    # Call the daily prediction function
    daily_prediction()
