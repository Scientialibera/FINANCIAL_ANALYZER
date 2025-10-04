"""
Main TSX Stock Analysis System
Complete pipeline: Download → Analyze → Simulate → Optimize
"""

import os
import sys
from tsx_data_downloader import TSXDataDownloader
from quant_analysis import QuantAnalyzer
from simulations import PortfolioSimulator
import pandas as pd

def main():
    print("=" * 80)
    print("TSX STOCK ANALYSIS SYSTEM")
    print("=" * 80)

    # Step 1: Download TSX stock data
    print("\n[STEP 1] DOWNLOADING TSX STOCK DATA")
    print("-" * 80)
    response = input("Download all TSX stock data? This may take a while. (y/n): ")

    if response.lower() == 'y':
        downloader = TSXDataDownloader(output_dir='data')
        downloader.download_all_tsx_stocks()
    else:
        print("Skipping download. Using existing data...")

    # Check if data directory exists
    if not os.path.exists('data') or len(os.listdir('data')) == 0:
        print("ERROR: No data found. Please download data first.")
        return

    # Step 2: Quantitative Analysis
    print("\n[STEP 2] QUANTITATIVE ANALYSIS")
    print("-" * 80)
    response = input("Run quantitative analysis on all stocks? (y/n): ")

    if response.lower() == 'y':
        analyzer = QuantAnalyzer(data_dir='data')
        results = analyzer.analyze_all_stocks('tsx_analysis.csv')

        # Display summary statistics
        print("\n" + "=" * 80)
        print("ANALYSIS SUMMARY")
        print("=" * 80)

        print(f"\nTotal stocks analyzed: {len(results)}")

        print("\n📈 TOP 10 STOCKS BY SHARPE RATIO:")
        print("-" * 80)
        top_sharpe = results.nlargest(10, 'sharpe_ratio')[
            ['ticker', 'sharpe_ratio', 'annual_return', 'annual_volatility', 'max_drawdown']
        ]
        print(top_sharpe.to_string(index=False))

        print("\n💰 TOP 10 STOCKS BY ANNUAL RETURN:")
        print("-" * 80)
        top_return = results.nlargest(10, 'annual_return')[
            ['ticker', 'annual_return', 'sharpe_ratio', 'annual_volatility', 'max_drawdown']
        ]
        print(top_return.to_string(index=False))

        print("\n🛡️ TOP 10 LOWEST VOLATILITY STOCKS:")
        print("-" * 80)
        low_vol = results.nsmallest(10, 'annual_volatility')[
            ['ticker', 'annual_volatility', 'annual_return', 'sharpe_ratio', 'max_drawdown']
        ]
        print(low_vol.to_string(index=False))

        print("\n📊 MARKET STATISTICS:")
        print("-" * 80)
        print(f"Average Annual Return: {results['annual_return'].mean():.2f}%")
        print(f"Median Annual Return: {results['annual_return'].median():.2f}%")
        print(f"Average Volatility: {results['annual_volatility'].mean():.2f}%")
        print(f"Average Sharpe Ratio: {results['sharpe_ratio'].mean():.3f}")
        print(f"Average Beta: {results['beta'].mean():.3f}")
        print(f"Average Max Drawdown: {results['max_drawdown'].mean():.2f}%")

        # Save results to file
        print(f"\n✓ Full results saved to: tsx_analysis.csv")

    # Step 3: Monte Carlo Simulations
    print("\n[STEP 3] MONTE CARLO SIMULATIONS")
    print("-" * 80)
    response = input("Run Monte Carlo simulation? (y/n): ")

    if response.lower() == 'y':
        ticker = input("Enter ticker symbol (e.g., RY.TO): ").upper()
        if not ticker.endswith('.TO'):
            ticker += '.TO'

        days = int(input("Forecast days ahead (default 252 = 1 year): ") or 252)
        sims = int(input("Number of simulations (default 10000): ") or 10000)

        simulator = PortfolioSimulator(data_dir='data')
        mc_results = simulator.monte_carlo_simulation(ticker, days=days, simulations=sims)

        if mc_results:
            print(f"\n📈 MONTE CARLO SIMULATION RESULTS: {ticker}")
            print("-" * 80)
            print(f"Current Price: ${mc_results['current_price']:.2f}")
            print(f"Simulations: {mc_results['simulations']:,}")
            print(f"Days Ahead: {mc_results['days_ahead']}")
            print(f"\nPredicted Price ({days} days):")
            print(f"  Mean: ${mc_results['mean_final_price']:.2f}")
            print(f"  Median: ${mc_results['median_final_price']:.2f}")
            print(f"  5th Percentile: ${mc_results['percentile_5']:.2f}")
            print(f"  95th Percentile: ${mc_results['percentile_95']:.2f}")
            print(f"  Min: ${mc_results['min_final_price']:.2f}")
            print(f"  Max: ${mc_results['max_final_price']:.2f}")
            print(f"\n✓ Probability of Profit: {mc_results['probability_profit']:.1f}%")

    # Step 4: Portfolio Optimization
    print("\n[STEP 4] PORTFOLIO OPTIMIZATION")
    print("-" * 80)
    response = input("Run portfolio optimization? (y/n): ")

    if response.lower() == 'y':
        print("\nEnter tickers separated by commas (e.g., RY.TO,TD.TO,BNS.TO)")
        print("Or press Enter for default Canadian bank portfolio")
        tickers_input = input("Tickers: ").strip()

        if tickers_input:
            tickers = [t.strip().upper() for t in tickers_input.split(',')]
            # Add .TO if not present
            tickers = [t if t.endswith('.TO') else t + '.TO' for t in tickers]
        else:
            tickers = ['RY.TO', 'TD.TO', 'BNS.TO', 'BMO.TO', 'CM.TO']

        print(f"\nOptimizing portfolio: {', '.join(tickers)}")

        simulator = PortfolioSimulator(data_dir='data')

        # Maximum Sharpe Ratio Portfolio
        print("\n💎 MAXIMUM SHARPE RATIO PORTFOLIO")
        print("-" * 80)
        max_sharpe = simulator.optimize_portfolio(tickers, objective='sharpe')
        print(f"Expected Annual Return: {max_sharpe['return']:.2f}%")
        print(f"Annual Volatility: {max_sharpe['volatility']:.2f}%")
        print(f"Sharpe Ratio: {max_sharpe['sharpe_ratio']:.3f}")
        print("\nOptimal Weights:")
        for ticker, weight in sorted(max_sharpe['weights'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {ticker:10s} {weight*100:6.2f}%")

        # Minimum Volatility Portfolio
        print("\n🛡️ MINIMUM VOLATILITY PORTFOLIO")
        print("-" * 80)
        min_vol = simulator.optimize_portfolio(tickers, objective='min_vol')
        print(f"Expected Annual Return: {min_vol['return']:.2f}%")
        print(f"Annual Volatility: {min_vol['volatility']:.2f}%")
        print(f"Sharpe Ratio: {min_vol['sharpe_ratio']:.3f}")
        print("\nOptimal Weights:")
        for ticker, weight in sorted(min_vol['weights'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {ticker:10s} {weight*100:6.2f}%")

        # Risk Parity Portfolio
        print("\n⚖️ RISK PARITY PORTFOLIO")
        print("-" * 80)
        risk_parity = simulator.risk_parity_portfolio(tickers)
        print(f"Expected Annual Return: {risk_parity['return']:.2f}%")
        print(f"Annual Volatility: {risk_parity['volatility']:.2f}%")
        print(f"Sharpe Ratio: {risk_parity['sharpe_ratio']:.3f}")
        print("\nOptimal Weights:")
        for ticker, weight in sorted(risk_parity['weights'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {ticker:10s} {weight*100:6.2f}%")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nGenerated Files:")
    print("  - data/ (historical stock data)")
    print("  - tsx_analysis.csv (quantitative metrics)")
    print("\nYou can run individual modules:")
    print("  - python tsx_data_downloader.py")
    print("  - python quant_analysis.py")
    print("  - python simulations.py")

if __name__ == "__main__":
    main()
