"""
Find the Best Balanced Portfolio across TSX sectors
Downloads data, analyzes stocks by sector, and optimizes a diversified 10-stock portfolio
"""

import yfinance as yf
import pandas as pd
import numpy as np
from tsx_data_downloader import TSXDataDownloader
from quant_analysis import QuantAnalyzer
from simulations import PortfolioSimulator

# Define TSX stocks by sector
TSX_SECTORS = {
    'Technology': [
        'SHOP.TO',   # Shopify
        'BB.TO',     # BlackBerry
        'LSPD.TO',   # Lightspeed
        'DOCN.TO',   # DigitalOcean (if available)
        'OTEX.TO',   # Open Text
        'ENGH.TO',   # Enghouse Systems
        'KXS.TO',    # Kinaxis
        'GDNP.TO',   # Goodfood Market
    ],
    'Financial': [
        'RY.TO',     # Royal Bank
        'TD.TO',     # TD Bank
        'BNS.TO',    # Scotiabank
        'BMO.TO',    # Bank of Montreal
        'CM.TO',     # CIBC
        'MFC.TO',    # Manulife
        'SLF.TO',    # Sun Life
        'POW.TO',    # Power Corporation
        'IFC.TO',    # Intact Financial
        'FFH.TO',    # Fairfax Financial
    ],
    'Energy': [
        'ENB.TO',    # Enbridge
        'TRP.TO',    # TC Energy
        'CNQ.TO',    # Canadian Natural Resources
        'SU.TO',     # Suncor
        'CVE.TO',    # Cenovus
        'IMO.TO',    # Imperial Oil
        'PPL.TO',    # Pembina Pipeline
        'ALA.TO',    # AltaGas
        'KEY.TO',    # Keyera
    ],
    'Industrial': [
        'CNR.TO',    # Canadian National Railway
        'CP.TO',     # Canadian Pacific Railway
        'WSP.TO',    # WSP Global
        'CCL.B.TO',  # CCL Industries
        'TIH.TO',    # Toromont Industries
        'STN.TO',    # Stantec
        'NFI.TO',    # NFI Group
        'BYD.TO',    # Boyd Group
    ],
    'Materials': [
        'ABX.TO',    # Barrick Gold
        'GOLD.TO',   # Barrick Gold (alternative ticker)
        'NTR.TO',    # Nutrien
        'FNV.TO',    # Franco-Nevada
        'WPM.TO',    # Wheaton Precious Metals
        'K.TO',      # Kinross Gold
        'FM.TO',     # First Quantum Minerals
        'CCO.TO',    # Cameco
        'WFG.TO',    # West Fraser Timber
    ],
    'Consumer': [
        'L.TO',      # Loblaw
        'ATD.TO',    # Alimentation Couche-Tard
        'DOL.TO',    # Dollarama
        'MG.TO',     # Magna International
        'QSR.TO',    # Restaurant Brands International
        'TFII.TO',   # TFI International
        'GIL.TO',    # Gildan Activewear
        'WN.TO',     # George Weston
    ],
    'Telecom': [
        'BCE.TO',    # Bell Canada
        'T.TO',      # Telus
        'RCI.B.TO',  # Rogers Communications
        'QBR.B.TO',  # Quebecor
    ],
    'Healthcare': [
        'CSU.TO',    # Constellation Software
        'WELL.TO',   # WELL Health Technologies
        'DOC.TO',    # CloudMD
        'PKI.TO',    # Parkland
    ],
    'Real Estate': [
        'BPY.UN.TO', # Brookfield Property
        'REI.UN.TO', # RioCan REIT
        'CHP.UN.TO', # Choice Properties REIT
        'SRU.UN.TO', # SmartCentres REIT
        'AP.UN.TO',  # Allied Properties REIT
    ],
    'Utilities': [
        'FTS.TO',    # Fortis
        'EMA.TO',    # Emera
        'H.TO',      # Hydro One
        'AQN.TO',    # Algonquin Power
        'CPX.TO',    # Capital Power
    ],
}

def download_sector_data():
    """Download data for all sector stocks"""
    print("=" * 80)
    print("DOWNLOADING TSX SECTOR DATA")
    print("=" * 80)

    downloader = TSXDataDownloader(output_dir='data')

    all_tickers = []
    for sector, tickers in TSX_SECTORS.items():
        all_tickers.extend(tickers)

    # Remove duplicates
    all_tickers = list(set(all_tickers))

    print(f"\nDownloading {len(all_tickers)} stocks across {len(TSX_SECTORS)} sectors...")

    successful = 0
    failed = 0

    for i, ticker in enumerate(all_tickers, 1):
        print(f"[{i}/{len(all_tickers)}] ", end='')
        result = downloader.download_stock_data(ticker)
        if result is not None:
            successful += 1
        else:
            failed += 1

    print(f"\nDownload complete: {successful} successful, {failed} failed")
    return all_tickers

def analyze_sectors():
    """Analyze each sector and find best stocks"""
    print("\n" + "=" * 80)
    print("ANALYZING SECTORS")
    print("=" * 80)

    analyzer = QuantAnalyzer(data_dir='data')

    # Download market data
    print("\nDownloading TSX Composite Index for beta/alpha...")
    market = yf.Ticker('^GSPTSE')
    market_data = market.history(period='max')

    sector_results = {}

    for sector, tickers in TSX_SECTORS.items():
        print(f"\n{sector}:")
        print("-" * 40)

        results = []
        for ticker in tickers:
            metrics = analyzer.analyze_stock(ticker, market_data)
            if metrics and not pd.isna(metrics.get('sharpe_ratio')):
                results.append(metrics)

        if results:
            df = pd.DataFrame(results)
            sector_results[sector] = df

            # Display top 3 by Sharpe ratio
            top3 = df.nlargest(3, 'sharpe_ratio')
            for _, stock in top3.iterrows():
                print(f"  {stock['ticker']:12s} Sharpe: {stock['sharpe_ratio']:6.3f}  "
                      f"Return: {stock['annual_return']:6.2f}%  "
                      f"Vol: {stock['annual_volatility']:5.2f}%")

    return sector_results

def select_best_balanced_portfolio(sector_results, stocks_per_sector=1):
    """
    Select the best balanced portfolio by picking top stocks from each sector
    """
    print("\n" + "=" * 80)
    print("BUILDING BALANCED PORTFOLIO")
    print("=" * 80)

    selected_stocks = []

    # Sort sectors by average Sharpe ratio to prioritize better sectors
    sector_quality = {}
    for sector, df in sector_results.items():
        if len(df) > 0:
            sector_quality[sector] = df['sharpe_ratio'].mean()

    sorted_sectors = sorted(sector_quality.items(), key=lambda x: x[1], reverse=True)

    print("\nSector Quality Ranking (by avg Sharpe):")
    for sector, avg_sharpe in sorted_sectors:
        print(f"  {sector:15s} {avg_sharpe:6.3f}")

    # Select top stock from each sector
    print("\nSelecting best stock from each sector:")
    for sector, _ in sorted_sectors:
        df = sector_results[sector]

        # Get top stock by Sharpe ratio
        top_stocks = df.nlargest(stocks_per_sector, 'sharpe_ratio')

        for _, stock in top_stocks.iterrows():
            selected_stocks.append({
                'ticker': stock['ticker'],
                'sector': sector,
                'sharpe_ratio': stock['sharpe_ratio'],
                'annual_return': stock['annual_return'],
                'annual_volatility': stock['annual_volatility'],
                'beta': stock['beta'],
                'max_drawdown': stock['max_drawdown']
            })
            print(f"  {sector:15s} -> {stock['ticker']:12s} "
                  f"(Sharpe: {stock['sharpe_ratio']:.3f}, "
                  f"Return: {stock['annual_return']:.2f}%)")

    return selected_stocks

def optimize_balanced_portfolio(selected_stocks, target_size=10):
    """
    Optimize the final portfolio selection and weights
    """
    print("\n" + "=" * 80)
    print(f"OPTIMIZING {target_size}-STOCK BALANCED PORTFOLIO")
    print("=" * 80)

    # If we have more than target_size, select top by Sharpe
    df = pd.DataFrame(selected_stocks)

    if len(df) > target_size:
        print(f"\nNarrowing down from {len(df)} stocks to {target_size}...")
        df = df.nlargest(target_size, 'sharpe_ratio')

    tickers = df['ticker'].tolist()

    print(f"\nFinal stock selection ({len(tickers)} stocks):")
    for _, stock in df.iterrows():
        print(f"  {stock['sector']:15s} {stock['ticker']:12s} "
              f"Sharpe: {stock['sharpe_ratio']:6.3f}  "
              f"Return: {stock['annual_return']:6.2f}%  "
              f"Vol: {stock['annual_volatility']:5.2f}%")

    # Optimize portfolio
    simulator = PortfolioSimulator(data_dir='data')

    print("\n" + "-" * 80)
    print("PORTFOLIO OPTIMIZATION RESULTS")
    print("-" * 80)

    # Strategy 1: Maximum Sharpe Ratio
    print("\n[1] MAXIMUM SHARPE RATIO PORTFOLIO")
    max_sharpe = simulator.optimize_portfolio(tickers, objective='sharpe')
    print(f"    Expected Return: {max_sharpe['return']:.2f}%")
    print(f"    Volatility: {max_sharpe['volatility']:.2f}%")
    print(f"    Sharpe Ratio: {max_sharpe['sharpe_ratio']:.3f}")
    print("    Weights:")
    for ticker, weight in sorted(max_sharpe['weights'].items(), key=lambda x: x[1], reverse=True):
        if weight > 0.01:
            sector = df[df['ticker'] == ticker]['sector'].values[0]
            print(f"      {ticker:12s} ({sector:15s}) {weight*100:5.2f}%")

    # Strategy 2: Risk Parity (equal risk contribution)
    print("\n[2] RISK PARITY PORTFOLIO")
    risk_parity = simulator.risk_parity_portfolio(tickers)
    print(f"    Expected Return: {risk_parity['return']:.2f}%")
    print(f"    Volatility: {risk_parity['volatility']:.2f}%")
    print(f"    Sharpe Ratio: {risk_parity['sharpe_ratio']:.3f}")
    print("    Weights:")
    for ticker, weight in sorted(risk_parity['weights'].items(), key=lambda x: x[1], reverse=True):
        if weight > 0.01:
            sector = df[df['ticker'] == ticker]['sector'].values[0]
            print(f"      {ticker:12s} ({sector:15s}) {weight*100:5.2f}%")

    # Strategy 3: Equal Weight (for comparison)
    print("\n[3] EQUAL WEIGHT PORTFOLIO")
    equal_weights = {ticker: 1/len(tickers) for ticker in tickers}
    equal_backtest = simulator.backtest_portfolio(equal_weights)
    print(f"    Expected Return: {equal_backtest['annual_return']:.2f}%")
    print(f"    Volatility: {equal_backtest['annual_volatility']:.2f}%")
    print(f"    Sharpe Ratio: {equal_backtest['sharpe_ratio']:.3f}")
    print("    Weights:")
    for ticker in tickers:
        sector = df[df['ticker'] == ticker]['sector'].values[0]
        print(f"      {ticker:12s} ({sector:15s}) {1/len(tickers)*100:5.2f}%")

    # Sector diversification analysis
    print("\n" + "-" * 80)
    print("SECTOR DIVERSIFICATION")
    print("-" * 80)

    for portfolio_name, weights in [
        ("Max Sharpe", max_sharpe['weights']),
        ("Risk Parity", risk_parity['weights']),
        ("Equal Weight", equal_weights)
    ]:
        sector_allocation = {}
        for ticker, weight in weights.items():
            sector = df[df['ticker'] == ticker]['sector'].values[0]
            sector_allocation[sector] = sector_allocation.get(sector, 0) + weight

        print(f"\n{portfolio_name}:")
        for sector, allocation in sorted(sector_allocation.items(), key=lambda x: x[1], reverse=True):
            if allocation > 0.01:
                print(f"  {sector:15s} {allocation*100:5.2f}%")

    return {
        'stocks': df,
        'max_sharpe': max_sharpe,
        'risk_parity': risk_parity,
        'equal_weight': equal_weights,
        'equal_weight_metrics': equal_backtest
    }

def main():
    print("=" * 80)
    print("TSX BALANCED PORTFOLIO FINDER")
    print("=" * 80)
    print("\nThis script will:")
    print("1. Download data for TSX stocks across all sectors")
    print("2. Analyze each sector to find best performers")
    print("3. Build a balanced 10-stock portfolio")
    print("4. Optimize portfolio weights using different strategies")

    # Step 1: Download data
    response = input("\nDownload sector data? (y/n, default=n): ").strip().lower()
    if response == 'y':
        download_sector_data()

    # Step 2: Analyze sectors
    sector_results = analyze_sectors()

    # Step 3: Select balanced portfolio
    selected = select_best_balanced_portfolio(sector_results, stocks_per_sector=1)

    # Step 4: Optimize
    results = optimize_balanced_portfolio(selected, target_size=10)

    # Summary
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    print("\nBest Balanced 10-Stock Portfolio:")
    print("\nUse MAXIMUM SHARPE RATIO allocation for best risk-adjusted returns:")
    print(f"  Expected Return: {results['max_sharpe']['return']:.2f}%")
    print(f"  Volatility: {results['max_sharpe']['volatility']:.2f}%")
    print(f"  Sharpe Ratio: {results['max_sharpe']['sharpe_ratio']:.3f}")

    print("\nPortfolio Allocation:")
    for ticker, weight in sorted(results['max_sharpe']['weights'].items(), key=lambda x: x[1], reverse=True):
        if weight > 0.01:
            sector = results['stocks'][results['stocks']['ticker'] == ticker]['sector'].values[0]
            print(f"  {weight*100:5.2f}%  {ticker:12s}  ({sector})")

    # Save results
    results['stocks'].to_csv('balanced_portfolio_stocks.csv', index=False)

    portfolio_summary = []
    for ticker, weight in results['max_sharpe']['weights'].items():
        stock = results['stocks'][results['stocks']['ticker'] == ticker].iloc[0]
        portfolio_summary.append({
            'ticker': ticker,
            'sector': stock['sector'],
            'weight_max_sharpe': weight * 100,
            'weight_risk_parity': results['risk_parity']['weights'][ticker] * 100,
            'weight_equal': results['equal_weight'][ticker] * 100,
            'sharpe_ratio': stock['sharpe_ratio'],
            'annual_return': stock['annual_return'],
            'annual_volatility': stock['annual_volatility'],
            'beta': stock['beta'],
            'max_drawdown': stock['max_drawdown']
        })

    pd.DataFrame(portfolio_summary).to_csv('balanced_portfolio_allocations.csv', index=False)

    print("\nResults saved:")
    print("  - balanced_portfolio_stocks.csv (stock details)")
    print("  - balanced_portfolio_allocations.csv (weight recommendations)")

if __name__ == "__main__":
    main()
