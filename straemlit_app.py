import streamlit as st
import pandas as pd
import plotly.express as px
from pmdarima import auto_arima
from prophet import Prophet
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

st.set_page_config(page_title="Stock Forecast Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("10000 stock price data.csv")
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    return df

df = load_data()

st.sidebar.title("📊 Dashboard Controls")
stock_options = df['Symbol'].unique()
selected_stock = st.sidebar.selectbox("Choose a stock:", stock_options)

stock_df = df[df['Symbol'] == selected_stock].copy()
stock_df.sort_values('Date', inplace=True)
min_date, max_date = stock_df['Date'].min(), stock_df['Date'].max()
date_range = st.sidebar.date_input("Select Date Range:", [min_date, max_date])

model_option = st.sidebar.radio("Forecast Model:", ["ARIMA", "SARIMA", "Prophet", "LSTM"])
forecast_period = st.sidebar.slider("Forecast Days:", 10, 100, 30)

stock_df['Date'] = pd.to_datetime(stock_df['Date'])
date_range = [pd.to_datetime(d) for d in date_range]  # optional
stock_df = stock_df[(stock_df['Date'] >= date_range[0]) & (stock_df['Date'] <= date_range[1])]



st.title(f"📈 {selected_stock} Dashboard")
st.markdown(f"Date Range: **{date_range[0]}** to **{date_range[1]}**")

fig_close = px.line(stock_df, x='Date', y='Close', title='Close Price')
st.plotly_chart(fig_close, use_container_width=True)

train_df = stock_df[['Date', 'Close']].copy()
train_df.set_index('Date', inplace=True)

def arima_forecast(train, period):
    model = auto_arima(train['Close'], seasonal=False)
    forecast = model.predict(n_periods=period)
    forecast_dates = pd.date_range(train.index[-1] + pd.Timedelta(days=1), periods=period)
    return pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast})

def sarima_forecast(train, period):
    model = auto_arima(train['Close'], seasonal=True, m=12)
    forecast = model.predict(n_periods=period)
    forecast_dates = pd.date_range(train.index[-1] + pd.Timedelta(days=1), periods=period)
    return pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast})

def prophet_forecast(train, period):
    df_prophet = train.reset_index().rename(columns={"Date": "ds", "Close": "y"})
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=period)
    forecast = model.predict(future)
    forecast = forecast[['ds', 'yhat']].tail(period)
    return forecast.rename(columns={'ds': 'Date', 'yhat': 'Forecast'})

def lstm_forecast(train, period):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(train[['Close']])

    seq_length = 10
    X, y = [], []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:i+seq_length])
        y.append(scaled_data[i+seq_length])
    X, y = np.array(X), np.array(y)

    train_size = int(0.8 * len(X))
    X_train, y_train = X[:train_size], y[:train_size]

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        LSTM(50),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

    last_seq = scaled_data[-seq_length:].reshape(1, seq_length, 1)
    preds = []
    for _ in range(period):
        pred = model.predict(last_seq, verbose=0)[0][0]
        preds.append(pred)
        last_seq = np.append(last_seq[:, 1:, :], [[[pred]]], axis=1)
    preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
    forecast_dates = pd.date_range(train.index[-1] + pd.Timedelta(days=1), periods=period)
    return pd.DataFrame({'Date': forecast_dates, 'Forecast': preds})

if model_option == "ARIMA":
    forecast_df = arima_forecast(train_df, forecast_period)
elif model_option == "SARIMA":
    forecast_df = sarima_forecast(train_df, forecast_period)
elif model_option == "Prophet":
    forecast_df = prophet_forecast(train_df, forecast_period)
else:
    forecast_df = lstm_forecast(train_df, forecast_period)

st.subheader(f"🔮 {model_option} Forecast for {forecast_period} Days")
fig_forecast = px.line(forecast_df, x='Date', y='Forecast', title=f"{model_option} Forecast")
st.plotly_chart(fig_forecast, use_container_width=True)

csv_data = pd.concat([stock_df.reset_index(), forecast_df]).to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download Data", data=csv_data, file_name=f"{selected_stock}_forecast.csv", mime='text/csv')
#.
#.\venv\Scripts\Activate
# pip install streamlit pandas plotly pmdarima prophet tensorflow scikit-learn
