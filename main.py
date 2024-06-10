import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator

# Step 1: Define the list of stock tickers
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

# Step 2: Fetch stock data
try:
    data = yf.download(tickers, period='1y', group_by='ticker')
except Exception as e:
    print(f"Error fetching stock data: {e}")
    exit(1)

# Step 3: Define a function to extract metrics
def get_metrics(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    try:
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        cashflow = stock.cashflow
    except Exception as e:
        print(f"Error fetching financials for {ticker}: {e}")
        return None

    metrics = {
        'Ticker': ticker,
        'PE Ratio': info.get('trailingPE', None),
        'EPS': info.get('trailingEps', None),
        'Dividend Yield': info.get('dividendYield', None),
        'Market Cap': info.get('marketCap', None),
        'Volume': info.get('volume', None),
        'Beta': info.get('beta', None),
        'ROE': info.get('returnOnEquity', None),
        'PEG Ratio': info.get('pegRatio', None)
    }
    return metrics

# Step 4: Collect metrics for each ticker
metrics_data = [get_metrics(ticker) for ticker in tickers]
metrics_data = [m for m in metrics_data if m is not None]

# Step 5: Convert to DataFrame
df_metrics = pd.DataFrame(metrics_data)

# Step 6: Set up the matplotlib figure for financial metrics
fig_metrics, axs_metrics = plt.subplots(3, 3, figsize=(15, 10))
axs_metrics = axs_metrics.flatten()

# Step 7: Create subplots for each metric
metrics = ['PE Ratio', 'EPS', 'Dividend Yield', 'Market Cap', 'Volume', 'Beta', 'ROE', 'PEG Ratio']

for i, metric in enumerate(metrics):
    sns.barplot(x='Ticker', y=metric, data=df_metrics, ax=axs_metrics[i])
    axs_metrics[i].set_title(metric)
    axs_metrics[i].set_xticklabels(axs_metrics[i].get_xticklabels(), rotation=45)

# Step 8: Add technical indicators
def plot_technical_indicators(ticker, data, ax):
    df = data[ticker]
    df['SMA_50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
    df['EMA_20'] = EMAIndicator(df['Close'], window=20).ema_indicator()
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()

    df[['Close', 'SMA_50', 'EMA_20']].plot(ax=ax[0])
    ax[0].set_title(f'Technical Indicators for {ticker}: SMA, EMA')
    ax[0].set_ylabel('Price')

    df['RSI'].plot(ax=ax[1], color='green')
    ax[1].axhline(70, color='red', linestyle='--')
    ax[1].axhline(30, color='blue', linestyle='--')
    ax[1].set_title('RSI')
    ax[1].set_ylabel('RSI Value')

# Plot technical indicators for each stock
fig_indicators, axs_indicators = plt.subplots(len(tickers), 2, figsize=(14, 5*len(tickers)))

for i, ticker in enumerate(tickers):
    plot_technical_indicators(ticker, data, axs_indicators[i])

fig_indicators.tight_layout()

# Step 9: Correlation heatmap
fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
corr = df_metrics.drop(columns=['Ticker']).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=.5, ax=ax_corr)
ax_corr.set_title('Correlation Heatmap of Financial Metrics')

# Step 10: Descriptive statistics for closing prices
closing_prices = pd.DataFrame({ticker: data[ticker]['Close'] for ticker in tickers})
summary_stats = closing_prices.describe()
print("\nDescriptive Statistics for Closing Prices")
print(summary_stats)

# Step 11: Monte Carlo Simulation
def monte_carlo_simulation(ticker, data, num_simulations=1000, num_days=252):
    df = data[ticker]['Close']
    log_returns = np.log(1 + df.pct_change().dropna())
    mean = log_returns.mean()
    std_dev = log_returns.std()

    simulations = np.zeros((num_days, num_simulations))
    last_price = df.iloc[-1]

    for i in range(num_simulations):
        daily_returns = np.random.normal(mean, std_dev, num_days)
        price_series = last_price * np.exp(np.cumsum(daily_returns))
        simulations[:, i] = price_series

    return simulations

# Monte Carlo simulations for each stock
fig_mc, axs_mc = plt.subplots(len(tickers), 1, figsize=(10, 6*len(tickers)))

for i, ticker in enumerate(tickers):
    simulations = monte_carlo_simulation(ticker, data)
    axs_mc[i].plot(simulations, lw=0.5, alpha=0.3)
    axs_mc[i].set_title(f'Monte Carlo Simulation: {ticker}')
    axs_mc[i].set_xlabel('Days')
    axs_mc[i].set_ylabel('Price')

fig_mc.tight_layout()

# Step 12: Multi-Factor Model
def multi_factor_model(tickers, factors, data):
    # Load data for the factors
    factor_data = yf.download(factors, period='1y')['Close']
    factor_data = factor_data.dropna()

    # Calculate returns
    returns = pd.concat([data[ticker]['Close'] for ticker in tickers] + [factor_data], axis=1).pct_change().dropna()
    returns.columns = tickers + factors

    # Separate stock returns and factor returns
    stock_returns = returns[tickers]
    factor_returns = returns[factors]

    # Run linear regression to find factor exposures
    exposures = {}
    for ticker in tickers:
        X = factor_returns
        y = stock_returns[ticker]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        exposures[ticker] = model.params

    return pd.DataFrame(exposures)

# Define factors
factors = ['^GSPC', 'IWM', 'MTUM', 'QUAL']  # S&P 500, Russell 2000, Momentum ETF, Quality ETF

# Multi-factor exposures
factor_exposures = multi_factor_model(tickers, factors, data)

# Plot factor exposures
fig_exposures, ax_exposures = plt.subplots(figsize=(12, 6))
factor_exposures.plot(kind='bar', ax=ax_exposures)
ax_exposures.set_title('Multi-Factor Model Exposures')
ax_exposures.set_ylabel('Exposure')
ax_exposures.set_xticklabels(ax_exposures.get_xticklabels(), rotation=0)
ax_exposures.legend(title='Tickers')

plt.tight_layout()
plt.show()
