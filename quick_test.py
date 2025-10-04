"""
Quick test of the complete system with a small sample
"""

from tsx_data_downloader import TSXDataDownloader
from quant_analysis import QuantAnalyzer
from simulations import PortfolioSimulator
import os

print("=" * 80)
print("QUICK TEST - TSX STOCK ANALYSIS SYSTEM")
print("=" * 80)

# Test with Canadian Big 5 Banks
test_tickers = ['RY.TO', 'TD.TO', 'BNS.TO', 'BMO.TO', 'CM.TO']

# Step 1: Download data for test tickers
print("\n[1] Downloading test data...")
print("-" * 80)
downloader = TSXDataDownloader(output_dir='data')

for ticker in test_tickers:
    downloader.download_stock_data(ticker)

# Step 2: Run quantitative analysis
print("\n[2] Running quantitative analysis...")
print("-" * 80)
analyzer = QuantAnalyzer(data_dir='data')

import yfinance as yf
market = yf.Ticker('^GSPTSE')
market_data = market.history(period='1y')  # Just 1 year for quick test

results = []
for ticker in test_tickers:
    print(f"Analyzing {ticker}...")
    metrics = analyzer.analyze_stock(ticker, market_data)
    if metrics:
        results.append(metrics)

# Display results
print("\n[3] Analysis Results:")
print("-" * 80)
import pandas as pd
df = pd.DataFrame(results)
print(df[['ticker', 'annual_return', 'annual_volatility', 'sharpe_ratio', 'beta', 'max_drawdown']].to_string(index=False))

# Step 3: Monte Carlo Simulation
print("\n[4] Monte Carlo Simulation (RY.TO)...")
print("-" * 80)
simulator = PortfolioSimulator(data_dir='data')
mc = simulator.monte_carlo_simulation('RY.TO', days=252, simulations=1000)

if mc:
    print(f"Current Price: ${mc['current_price']:.2f}")
    print(f"Predicted (1 year): ${mc['mean_final_price']:.2f} (mean), ${mc['median_final_price']:.2f} (median)")
    print(f"95% Confidence: ${mc['percentile_5']:.2f} - ${mc['percentile_95']:.2f}")
    print(f"Probability of Profit: {mc['probability_profit']:.1f}%")

# Step 4: Portfolio Optimization
print("\n[5] Portfolio Optimization...")
print("-" * 80)

# Maximum Sharpe
print("\nMaximum Sharpe Ratio Portfolio:")
max_sharpe = simulator.optimize_portfolio(test_tickers, objective='sharpe')
print(f"Expected Return: {max_sharpe['return']:.2f}%")
print(f"Volatility: {max_sharpe['volatility']:.2f}%")
print(f"Sharpe Ratio: {max_sharpe['sharpe_ratio']:.3f}")
print("Weights:")
for ticker, weight in sorted(max_sharpe['weights'].items(), key=lambda x: x[1], reverse=True):
    if weight > 0.01:  # Only show weights > 1%
        print(f"  {ticker}: {weight*100:.1f}%")

# Risk Parity
print("\nRisk Parity Portfolio:")
risk_parity = simulator.risk_parity_portfolio(test_tickers)
print(f"Expected Return: {risk_parity['return']:.2f}%")
print(f"Volatility: {risk_parity['volatility']:.2f}%")
print(f"Sharpe Ratio: {risk_parity['sharpe_ratio']:.3f}")
print("Weights:")
for ticker, weight in sorted(risk_parity['weights'].items(), key=lambda x: x[1], reverse=True):
    if weight > 0.01:
        print(f"  {ticker}: {weight*100:.1f}%")

print("\n" + "=" * 80)
print("TEST COMPLETE! All systems working.")
print("=" * 80)
print("\nTo run full analysis:")
print("  python main.py")
