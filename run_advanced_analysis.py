#!/usr/bin/env python
"""
Advanced TSX Analysis - Multiple Portfolio Strategies with Complex Simulations
Runs comprehensive quantitative analysis on all downloaded TSX stocks
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import yfinance as yf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import DIRS, ANALYSIS
from src.analyzer import QuantAnalyzer
from src.portfolio_optimizer import PortfolioSimulator
from src.monte_carlo import AdvancedSimulator

print("="*80)
print("ADVANCED TSX ANALYSIS - MULTIPLE PORTFOLIO STRATEGIES")
print("Complex Simulations & Quantitative Methods")
print("="*80)

# Step 1: Load and analyze all stocks
print("\n[1/5] Loading and analyzing all TSX stocks...")
print("-"*80)

analyzer = QuantAnalyzer(data_dir=str(DIRS['data']))

# Download market benchmark
print("Downloading TSX Composite Index...")
market = yf.Ticker('^GSPTSE')
market_data = market.history(period='max')

# Get all stocks
import os
stock_files = [f for f in os.listdir(DIRS['data']) if f.endswith('_history.csv')]
tickers = [f.replace('_history.csv', '').replace('_', '.') for f in stock_files]

print(f"Found {len(tickers)} stocks with downloaded data")
print("Analyzing with 20+ quantitative metrics...")

results = []
for i, ticker in enumerate(tickers, 1):
    if i % 25 == 0:
        print(f"  Progress: {i}/{len(tickers)}")

    metrics = analyzer.analyze_stock(ticker, market_data)
    if metrics and not pd.isna(metrics.get('sharpe_ratio')):
        results.append(metrics)

df_all = pd.DataFrame(results)

# Save complete analysis
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
analysis_file = DIRS['reports'] / f'complete_analysis_{timestamp}.csv'
df_all.to_csv(analysis_file, index=False)

print(f"\nAnalyzed {len(df_all)} stocks")
print(f"Results saved to: {analysis_file}")

# Display market statistics
print("\nMARKET STATISTICS:")
print(f"  Average Annual Return: {df_all['annual_return'].mean():.2f}%")
print(f"  Average Sharpe Ratio: {df_all['sharpe_ratio'].mean():.3f}")
print(f"  Stocks with Sharpe > 1.0: {len(df_all[df_all['sharpe_ratio'] > 1.0])}")
print(f"  Stocks with Sharpe > 0.5: {len(df_all[df_all['sharpe_ratio'] > 0.5])}")

# Step 2: Build multiple portfolio strategies
print("\n[2/5] Building Multiple Portfolio Strategies...")
print("-"*80)

# Filter quality stocks
quality = df_all[
    (df_all['sharpe_ratio'] > 0.3) &
    (df_all['sharpe_ratio'].notna()) &
    (df_all['beta'].notna()) &
    (df_all['days_of_data'] >= 250)
].copy()

print(f"Filtered to {len(quality)} quality stocks (Sharpe > 0.3, sufficient data)")

simulator = PortfolioSimulator(data_dir=str(DIRS['data']))

# Strategy 1: Top 10 by Sharpe (Aggressive Growth)
print("\n[Strategy 1] Aggressive Growth - Top 10 Sharpe")
top10_sharpe = quality.nlargest(10, 'sharpe_ratio')
tickers_s1 = top10_sharpe['ticker'].tolist()
portfolio_s1 = simulator.optimize_portfolio(tickers_s1, objective='sharpe')
print(f"  Return: {portfolio_s1['return']:.2f}%, Volatility: {portfolio_s1['volatility']:.2f}%, Sharpe: {portfolio_s1['sharpe_ratio']:.3f}")

# Strategy 2: Top 15 by Sharpe (Balanced Growth)
print("\n[Strategy 2] Balanced Growth - Top 15 Sharpe")
top15_sharpe = quality.nlargest(15, 'sharpe_ratio')
tickers_s2 = top15_sharpe['ticker'].tolist()
portfolio_s2 = simulator.optimize_portfolio(tickers_s2, objective='sharpe')
print(f"  Return: {portfolio_s2['return']:.2f}%, Volatility: {portfolio_s2['volatility']:.2f}%, Sharpe: {portfolio_s2['sharpe_ratio']:.3f}")

# Strategy 3: Top 20 by Sharpe (Diversified)
print("\n[Strategy 3] Diversified - Top 20 Sharpe")
top20_sharpe = quality.nlargest(20, 'sharpe_ratio')
tickers_s3 = top20_sharpe['ticker'].tolist()
portfolio_s3 = simulator.optimize_portfolio(tickers_s3, objective='sharpe')
print(f"  Return: {portfolio_s3['return']:.2f}%, Volatility: {portfolio_s3['volatility']:.2f}%, Sharpe: {portfolio_s3['sharpe_ratio']:.3f}")

# Strategy 4: High Return (>20%), Top Sharpe (Momentum)
print("\n[Strategy 4] High Momentum - Return > 20%, Top Sharpe")
high_return = quality[quality['annual_return'] > 20].nlargest(15, 'sharpe_ratio')
if len(high_return) >= 5:
    tickers_s4 = high_return['ticker'].tolist()
    portfolio_s4 = simulator.optimize_portfolio(tickers_s4, objective='sharpe')
    print(f"  Return: {portfolio_s4['return']:.2f}%, Volatility: {portfolio_s4['volatility']:.2f}%, Sharpe: {portfolio_s4['sharpe_ratio']:.3f}")
else:
    portfolio_s4 = None
    print("  Insufficient stocks (need 5+)")

# Strategy 5: Low Volatility (<30%), Best Sharpe (Conservative)
print("\n[Strategy 5] Conservative - Low Volatility, Best Sharpe")
low_vol = quality[quality['annual_volatility'] < 30].nlargest(12, 'sharpe_ratio')
if len(low_vol) >= 5:
    tickers_s5 = low_vol['ticker'].tolist()
    portfolio_s5 = simulator.optimize_portfolio(tickers_s5, objective='sharpe')
    print(f"  Return: {portfolio_s5['return']:.2f}%, Volatility: {portfolio_s5['volatility']:.2f}%, Sharpe: {portfolio_s5['sharpe_ratio']:.3f}")
else:
    portfolio_s5 = None
    print("  Insufficient stocks")

# Strategy 6: Risk Parity on Top 20
print("\n[Strategy 6] Risk Parity - Equal Risk Contribution")
portfolio_s6 = simulator.risk_parity_portfolio(tickers_s3)
print(f"  Return: {portfolio_s6['return']:.2f}%, Volatility: {portfolio_s6['volatility']:.2f}%, Sharpe: {portfolio_s6['sharpe_ratio']:.3f}")

# Strategy 7: Minimum Volatility on Top 20
print("\n[Strategy 7] Minimum Volatility - Lowest Risk")
portfolio_s7 = simulator.optimize_portfolio(tickers_s3, objective='min_vol')
print(f"  Return: {portfolio_s7['return']:.2f}%, Volatility: {portfolio_s7['volatility']:.2f}%, Sharpe: {portfolio_s7['sharpe_ratio']:.3f}")

# Step 3: Monte Carlo Simulations on Top Stocks
print("\n[3/5] Running Monte Carlo Simulations on Top 5 Stocks...")
print("-"*80)

mc_simulator = AdvancedSimulator(data_dir=str(DIRS['data']))

top5_stocks = quality.nlargest(5, 'sharpe_ratio')['ticker'].tolist()
mc_results = []

for ticker in top5_stocks:
    print(f"\nSimulating {ticker}...")

    # Run regime switching simulation
    result = mc_simulator.regime_switching_simulation(ticker, days=252, simulations=5000)

    if result:
        print(f"  Current: ${result['current_price']:.2f}")
        print(f"  1-Year Forecast: ${result['median_final_price']:.2f}")
        print(f"  Probability of Profit: {result['probability_profit']:.1f}%")
        print(f"  Expected Return: {result['expected_return']:.2f}%")

        mc_results.append({
            'ticker': ticker,
            'current_price': result['current_price'],
            'forecast_price': result['median_final_price'],
            'prob_profit': result['probability_profit'],
            'expected_return': result['expected_return'],
            'var_95': result['var_95'],
            'cvar_95': result['cvar_95']
        })

# Save Monte Carlo results
mc_df = pd.DataFrame(mc_results)
mc_file = DIRS['reports'] / f'monte_carlo_top5_{timestamp}.csv'
mc_df.to_csv(mc_file, index=False)
print(f"\nMonte Carlo results saved to: {mc_file}")

# Step 4: Save all portfolio strategies
print("\n[4/5] Saving Portfolio Strategies...")
print("-"*80)

portfolios = {
    'Aggressive_Growth_10': (portfolio_s1, tickers_s1),
    'Balanced_Growth_15': (portfolio_s2, tickers_s2),
    'Diversified_20': (portfolio_s3, tickers_s3),
    'High_Momentum': (portfolio_s4, tickers_s4) if portfolio_s4 else (None, []),
    'Conservative_Low_Vol': (portfolio_s5, tickers_s5) if portfolio_s5 else (None, []),
    'Risk_Parity_20': (portfolio_s6, tickers_s3),
    'Minimum_Volatility_20': (portfolio_s7, tickers_s3),
}

all_portfolios = []

for strategy_name, (portfolio, tickers_list) in portfolios.items():
    if portfolio is None:
        continue

    for ticker in tickers_list:
        stock_data = df_all[df_all['ticker'] == ticker].iloc[0]
        all_portfolios.append({
            'strategy': strategy_name,
            'ticker': ticker,
            'weight_pct': portfolio['weights'].get(ticker, 0) * 100,
            'sharpe_ratio': stock_data['sharpe_ratio'],
            'annual_return': stock_data['annual_return'],
            'annual_volatility': stock_data['annual_volatility'],
            'beta': stock_data['beta'],
            'alpha': stock_data['alpha'],
            'max_drawdown': stock_data['max_drawdown'],
            'portfolio_return': portfolio['return'],
            'portfolio_volatility': portfolio['volatility'],
            'portfolio_sharpe': portfolio['sharpe_ratio']
        })

portfolio_df = pd.DataFrame(all_portfolios)
portfolio_file = DIRS['portfolios'] / f'all_strategies_{timestamp}.csv'
portfolio_df.to_csv(portfolio_file, index=False)

print(f"All portfolios saved to: {portfolio_file}")

# Step 5: Generate comprehensive report
print("\n[5/5] Generating Comprehensive Report...")
print("-"*80)

report = []
report.append("="*80)
report.append("TSX ADVANCED ANALYSIS - COMPREHENSIVE REPORT")
report.append("="*80)
report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report.append(f"Total Stocks Analyzed: {len(df_all)}")
report.append("")

report.append("MARKET STATISTICS")
report.append("-"*80)
report.append(f"Average Annual Return: {df_all['annual_return'].mean():.2f}%")
report.append(f"Median Annual Return: {df_all['annual_return'].median():.2f}%")
report.append(f"Average Sharpe Ratio: {df_all['sharpe_ratio'].mean():.3f}")
report.append(f"Median Sharpe Ratio: {df_all['sharpe_ratio'].median():.3f}")
report.append(f"Average Volatility: {df_all['annual_volatility'].mean():.2f}%")
report.append(f"Average Beta: {df_all['beta'].mean():.3f}")
report.append(f"Average Max Drawdown: {df_all['max_drawdown'].mean():.2f}%")
report.append("")

report.append("TOP 10 STOCKS BY SHARPE RATIO")
report.append("-"*80)
top10 = df_all.nlargest(10, 'sharpe_ratio')[['ticker', 'sharpe_ratio', 'annual_return', 'annual_volatility']]
for _, row in top10.iterrows():
    report.append(f"{row['ticker']:12s}  Sharpe: {row['sharpe_ratio']:6.3f}  Return: {row['annual_return']:6.2f}%  Vol: {row['annual_volatility']:5.2f}%")
report.append("")

report.append("PORTFOLIO STRATEGIES COMPARISON")
report.append("-"*80)
report.append(f"{'Strategy':<30s} {'Return':>8s} {'Vol':>8s} {'Sharpe':>8s}")
report.append("-"*80)

for strategy_name, (portfolio, _) in portfolios.items():
    if portfolio:
        report.append(f"{strategy_name:<30s} {portfolio['return']:>7.2f}% {portfolio['volatility']:>7.2f}% {portfolio['sharpe_ratio']:>8.3f}")

report.append("")
report.append("RECOMMENDED PORTFOLIO: " + max(
    [(name, port['sharpe_ratio']) for name, (port, _) in portfolios.items() if port],
    key=lambda x: x[1]
)[0])
report.append("")

report.append("FILES GENERATED")
report.append("-"*80)
report.append(f"Complete Analysis: {analysis_file.name}")
report.append(f"All Portfolios: {portfolio_file.name}")
report.append(f"Monte Carlo: {mc_file.name}")
report.append("")

# Save report
report_file = DIRS['reports'] / f'comprehensive_report_{timestamp}.txt'
with open(report_file, 'w') as f:
    f.write('\n'.join(report))

# Print report
print('\n'.join(report))

print("="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\nAll files saved in: {DIRS['output']}")
print(f"Reports: {DIRS['reports']}")
print(f"Portfolios: {DIRS['portfolios']}")
