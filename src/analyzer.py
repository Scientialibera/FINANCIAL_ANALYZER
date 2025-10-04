"""
Quantitative Analysis for TSX Stocks
Includes: Volatility, Alpha, Beta, Sharpe Ratio, Sortino Ratio,
Maximum Drawdown, VaR, CVaR, Information Ratio, and more
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import os
import json
import warnings
warnings.filterwarnings('ignore')

class QuantAnalyzer:
    def __init__(self, data_dir='data', benchmark_ticker='^GSPTSE'):
        """
        Initialize analyzer
        data_dir: directory containing downloaded stock data
        benchmark_ticker: TSX Composite Index (^GSPTSE)
        """
        self.data_dir = data_dir
        self.benchmark_ticker = benchmark_ticker
        self.risk_free_rate = 0.045  # Current Canadian risk-free rate (approx)

    def load_stock_data(self, ticker):
        """Load historical data for a stock"""
        csv_path = os.path.join(self.data_dir, f"{ticker.replace('.', '_')}_history.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            return df
        return None

    def calculate_returns(self, prices):
        """Calculate daily returns"""
        return prices.pct_change().dropna()

    def calculate_volatility(self, returns, annualize=True):
        """Calculate volatility (standard deviation of returns)"""
        vol = returns.std()
        if annualize:
            vol *= np.sqrt(252)  # 252 trading days per year
        return vol

    def calculate_sharpe_ratio(self, returns, risk_free_rate=None):
        """
        Calculate Sharpe Ratio
        (Mean Return - Risk Free Rate) / Standard Deviation
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate

        # Annualize returns
        annual_return = returns.mean() * 252
        annual_vol = self.calculate_volatility(returns, annualize=True)

        sharpe = (annual_return - risk_free_rate) / annual_vol
        return sharpe

    def calculate_sortino_ratio(self, returns, risk_free_rate=None):
        """
        Calculate Sortino Ratio
        Similar to Sharpe but only considers downside volatility
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate

        annual_return = returns.mean() * 252
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252)

        if downside_vol == 0:
            return np.nan

        sortino = (annual_return - risk_free_rate) / downside_vol
        return sortino

    def calculate_beta(self, stock_returns, market_returns):
        """
        Calculate Beta (systematic risk)
        Beta = Covariance(stock, market) / Variance(market)
        """
        # Align the data
        combined = pd.DataFrame({'stock': stock_returns, 'market': market_returns}).dropna()

        if len(combined) < 30:  # Need enough data points
            return np.nan

        covariance = combined['stock'].cov(combined['market'])
        market_variance = combined['market'].var()

        beta = covariance / market_variance
        return beta

    def calculate_alpha(self, stock_returns, market_returns, beta=None):
        """
        Calculate Alpha (excess return)
        Alpha = Stock Return - (Risk Free Rate + Beta * (Market Return - Risk Free Rate))
        """
        if beta is None:
            beta = self.calculate_beta(stock_returns, market_returns)

        stock_return = stock_returns.mean() * 252
        market_return = market_returns.mean() * 252

        expected_return = self.risk_free_rate + beta * (market_return - self.risk_free_rate)
        alpha = stock_return - expected_return

        return alpha

    def calculate_max_drawdown(self, prices):
        """
        Calculate Maximum Drawdown
        Maximum loss from peak to trough
        """
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        return max_dd

    def calculate_var(self, returns, confidence=0.95):
        """
        Calculate Value at Risk (VaR)
        Maximum loss at given confidence level
        """
        var = returns.quantile(1 - confidence)
        return var

    def calculate_cvar(self, returns, confidence=0.95):
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall
        Average loss beyond VaR threshold
        """
        var = self.calculate_var(returns, confidence)
        cvar = returns[returns <= var].mean()
        return cvar

    def calculate_information_ratio(self, stock_returns, benchmark_returns):
        """
        Calculate Information Ratio
        Measures excess return per unit of tracking error
        """
        combined = pd.DataFrame({'stock': stock_returns, 'bench': benchmark_returns}).dropna()

        if len(combined) < 30:
            return np.nan

        excess_returns = combined['stock'] - combined['bench']
        tracking_error = excess_returns.std() * np.sqrt(252)

        if tracking_error == 0:
            return np.nan

        ir = (excess_returns.mean() * 252) / tracking_error
        return ir

    def calculate_calmar_ratio(self, returns, prices):
        """
        Calculate Calmar Ratio
        Annual return / Maximum Drawdown
        """
        annual_return = returns.mean() * 252
        max_dd = abs(self.calculate_max_drawdown(prices))

        if max_dd == 0:
            return np.nan

        calmar = annual_return / max_dd
        return calmar

    def calculate_skewness(self, returns):
        """Calculate skewness of return distribution"""
        return stats.skew(returns.dropna())

    def calculate_kurtosis(self, returns):
        """Calculate kurtosis of return distribution"""
        return stats.kurtosis(returns.dropna())

    def calculate_downside_deviation(self, returns, mar=0):
        """
        Calculate Downside Deviation
        mar: Minimum Acceptable Return
        """
        downside_returns = returns[returns < mar]
        downside_dev = downside_returns.std() * np.sqrt(252)
        return downside_dev

    def calculate_ulcer_index(self, prices):
        """
        Calculate Ulcer Index
        Measures depth and duration of drawdowns
        """
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.cummax()
        drawdown = 100 * ((cumulative - running_max) / running_max)
        ulcer = np.sqrt((drawdown ** 2).mean())
        return ulcer

    def analyze_stock(self, ticker, market_data=None):
        """
        Perform comprehensive quantitative analysis on a stock
        """
        # Load stock data
        df = self.load_stock_data(ticker)
        if df is None or df.empty:
            print(f"No data found for {ticker}")
            return None

        # Calculate returns
        prices = df['Close']
        returns = self.calculate_returns(prices)

        # Load or calculate market returns for beta/alpha
        market_returns = None
        if market_data is not None:
            market_returns = self.calculate_returns(market_data['Close'])

        # Calculate all metrics
        metrics = {
            'ticker': ticker,
            'start_date': str(df.index[0].date()),
            'end_date': str(df.index[-1].date()),
            'days_of_data': len(df),

            # Returns
            'total_return': ((prices.iloc[-1] / prices.iloc[0]) - 1) * 100,
            'annual_return': returns.mean() * 252 * 100,
            'daily_return_mean': returns.mean() * 100,
            'daily_return_median': returns.median() * 100,

            # Volatility
            'daily_volatility': self.calculate_volatility(returns, annualize=False) * 100,
            'annual_volatility': self.calculate_volatility(returns, annualize=True) * 100,
            'downside_deviation': self.calculate_downside_deviation(returns) * 100,

            # Risk-Adjusted Returns
            'sharpe_ratio': self.calculate_sharpe_ratio(returns),
            'sortino_ratio': self.calculate_sortino_ratio(returns),

            # Drawdown
            'max_drawdown': self.calculate_max_drawdown(prices) * 100,
            'calmar_ratio': self.calculate_calmar_ratio(returns, prices),
            'ulcer_index': self.calculate_ulcer_index(prices),

            # Risk Metrics
            'var_95': self.calculate_var(returns, 0.95) * 100,
            'cvar_95': self.calculate_cvar(returns, 0.95) * 100,
            'var_99': self.calculate_var(returns, 0.99) * 100,
            'cvar_99': self.calculate_cvar(returns, 0.99) * 100,

            # Distribution
            'skewness': self.calculate_skewness(returns),
            'kurtosis': self.calculate_kurtosis(returns),

            # Current Stats
            'current_price': prices.iloc[-1],
            'high_52w': df['High'].iloc[-252:].max() if len(df) >= 252 else df['High'].max(),
            'low_52w': df['Low'].iloc[-252:].min() if len(df) >= 252 else df['Low'].min(),
        }

        # Add market-relative metrics if market data available
        if market_returns is not None:
            beta = self.calculate_beta(returns, market_returns)
            metrics['beta'] = beta
            metrics['alpha'] = self.calculate_alpha(returns, market_returns, beta) * 100
            metrics['information_ratio'] = self.calculate_information_ratio(returns, market_returns)
        else:
            metrics['beta'] = None
            metrics['alpha'] = None
            metrics['information_ratio'] = None

        return metrics

    def analyze_all_stocks(self, output_file='tsx_analysis.csv'):
        """
        Analyze all stocks in the data directory
        """
        import yfinance as yf

        # Download market benchmark data
        print("Downloading TSX Composite Index data for beta/alpha calculations...")
        market = yf.Ticker(self.benchmark_ticker)
        market_data = market.history(period='max')

        # Get all stock files
        stock_files = [f for f in os.listdir(self.data_dir) if f.endswith('_history.csv')]
        tickers = [f.replace('_history.csv', '').replace('_', '.') for f in stock_files]

        print(f"Analyzing {len(tickers)} stocks...")
        print("=" * 80)

        results = []
        for i, ticker in enumerate(tickers, 1):
            print(f"[{i}/{len(tickers)}] Analyzing {ticker}...", end=' ')
            metrics = self.analyze_stock(ticker, market_data)
            if metrics:
                results.append(metrics)
                print("OK")
            else:
                print("FAILED")

        # Create DataFrame
        df_results = pd.DataFrame(results)

        # Save to CSV
        df_results.to_csv(output_file, index=False)
        print("=" * 80)
        print(f"Analysis complete! Results saved to: {output_file}")

        return df_results

if __name__ == "__main__":
    analyzer = QuantAnalyzer(data_dir='data')
    results = analyzer.analyze_all_stocks('tsx_analysis.csv')

    # Display top performers by Sharpe Ratio
    print("\nTop 10 Stocks by Sharpe Ratio:")
    print(results.nlargest(10, 'sharpe_ratio')[['ticker', 'sharpe_ratio', 'annual_return', 'annual_volatility']])

    print("\nTop 10 Stocks by Annual Return:")
    print(results.nlargest(10, 'annual_return')[['ticker', 'annual_return', 'sharpe_ratio', 'max_drawdown']])
