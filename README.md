# Time-Series-Analysis-of-Stock-Market
Project Overview

This project focuses on forecasting stock prices of four major Indian companies (TCS, ULTRACEMCO, HDFC, RELIANCE) using multiple time series models. The goal is to analyze trend, seasonality, and volatility patterns in financial data and evaluate the performance of different forecasting techniques for risk management and investment decision-making.

The models implemented include:

1. ARIMA & SARIMA – for capturing trend and seasonality in stable stocks.

2. Prophet (by Facebook/Meta) – for handling flexible seasonality and holiday effects.

3. LSTM (Long Short-Term Memory) – for capturing non-linear and volatile price movements.

4. Baseline Models (Moving Averages) – for simple smoothing and trend tracking.

In addition, a Power BI dashboard was developed to interactively visualize:
Stock trends, daily returns, and moving averages
Forecasted vs. actual prices
Key KPIs (total traded volume, average close price, max high, min low, deliverable percentage)

Key Insights

Decomposition revealed clear trend and seasonal patterns in certain stocks (e.g., ULTRACEMCO), while others showed high volatility (e.g., TCS).

RMSE evaluation showed ARIMA/SARIMA performed best on stable stocks, while LSTM outperformed on volatile stocks, highlighting the need to match models to stock behavior.

The Power BI dashboard enabled risk-focused monitoring with interactive insights for analysts.
