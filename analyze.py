#!/usr/bin/env python
"""
TSX Financial Analyzer - Main Executable
Enterprise-grade quantitative analysis system for TSX stocks

Usage:
    python analyze.py --full          # Complete analysis (scrape + download + analyze + optimize)
    python analyze.py --quick         # Quick test with sample stocks
    python analyze.py --portfolio     # Portfolio optimization only
    python analyze.py --simulate TICKER  # Monte Carlo simulation for specific ticker

Examples:
    python analyze.py --full
    python analyze.py --quick --stocks 50
    python analyze.py --portfolio --num-stocks 10
    python analyze.py --simulate RY.TO --days 252
"""

import sys
import argparse
import logging
import logging.config
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import DIRS, ANALYSIS, PORTFOLIO, SIMULATION, LOGGING, get_output_filename
from src.ticker_scraper import TSXTickerScraper
from src.data_downloader import TSXDataDownloader
from src.analyzer import QuantAnalyzer
from src.portfolio_optimizer import PortfolioSimulator
from src.monte_carlo import AdvancedSimulator

# Setup logging
logging.config.dictConfig(LOGGING)
logger = logging.getLogger(__name__)

class TSXAnalyzer:
    """Main analyzer class orchestrating all components"""

    def __init__(self):
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results = {}

    def log_start(self, task):
        """Log task start"""
        logger.info("="*80)
        logger.info(f"STARTING: {task}")
        logger.info(f"Run ID: {self.run_id}")
        logger.info("="*80)

    def log_complete(self, task, details=None):
        """Log task completion"""
        logger.info(f"COMPLETED: {task}")
        if details:
            for key, value in details.items():
                logger.info(f"  {key}: {value}")

    def scrape_tickers(self):
        """Scrape all TSX tickers"""
        self.log_start("Ticker Scraping")

        scraper = TSXTickerScraper()
        tickers = scraper.get_all_tsx_tickers(validate=True, validation_sample=30)

        # Save to cache
        import pandas as pd
        ticker_file = DIRS['cache'] / f'tsx_tickers_{self.run_id}.csv'
        pd.DataFrame(tickers).to_csv(ticker_file, index=False)

        self.results['tickers'] = tickers
        self.results['ticker_file'] = str(ticker_file)

        self.log_complete("Ticker Scraping", {
            'Total Tickers': len(tickers),
            'Saved To': ticker_file
        })

        return tickers

    def download_data(self, tickers=None, limit=None):
        """Download historical data"""
        self.log_start("Data Download")

        if tickers is None:
            # Load from most recent cache
            import pandas as pd
            ticker_files = list(DIRS['cache'].glob('tsx_tickers_*.csv'))
            if ticker_files:
                latest = max(ticker_files, key=lambda p: p.stat().st_mtime)
                df = pd.read_csv(latest)
                tickers = df['yahoo_ticker'].tolist()
                logger.info(f"Loaded {len(tickers)} tickers from cache: {latest.name}")
            else:
                logger.warning("No cached tickers found, scraping...")
                ticker_data = self.scrape_tickers()
                tickers = [t['yahoo_ticker'] for t in ticker_data]

        if limit:
            tickers = tickers[:limit]
            logger.info(f"Limited to first {limit} stocks")

        downloader = TSXDataDownloader(output_dir=str(DIRS['data']))

        successful = []
        failed = []

        for i, ticker in enumerate(tickers, 1):
            logger.info(f"[{i}/{len(tickers)}] Downloading {ticker}...")
            result = downloader.download_stock_data(ticker, period='max')

            if result is not None and len(result) >= ANALYSIS['min_data_points']:
                successful.append(ticker)
            else:
                failed.append(ticker)

        # Save download log
        import pandas as pd
        log_file = DIRS['logs'] / f'download_log_{self.run_id}.csv'
        pd.DataFrame({
            'successful': pd.Series(successful),
            'failed': pd.Series(failed)
        }).to_csv(log_file, index=False)

        self.results['downloaded'] = successful
        self.results['download_log'] = str(log_file)

        self.log_complete("Data Download", {
            'Successful': len(successful),
            'Failed': len(failed),
            'Success Rate': f"{len(successful)/len(tickers)*100:.1f}%",
            'Log File': log_file
        })

        return successful

    def analyze_stocks(self):
        """Perform quantitative analysis"""
        self.log_start("Quantitative Analysis")

        import yfinance as yf
        import pandas as pd

        analyzer = QuantAnalyzer(data_dir=str(DIRS['data']))

        # Download market data
        logger.info(f"Downloading benchmark: {ANALYSIS['benchmark_ticker']}")
        market = yf.Ticker(ANALYSIS['benchmark_ticker'])
        market_data = market.history(period='max')

        # Get all stock files
        import os
        stock_files = [f for f in os.listdir(DIRS['data']) if f.endswith('_history.csv')]
        tickers = [f.replace('_history.csv', '').replace('_', '.') for f in stock_files]

        logger.info(f"Analyzing {len(tickers)} stocks...")

        results = []
        for i, ticker in enumerate(tickers, 1):
            if i % 25 == 0:
                logger.info(f"Progress: {i}/{len(tickers)}")

            metrics = analyzer.analyze_stock(ticker, market_data)
            if metrics and not pd.isna(metrics.get('sharpe_ratio')):
                results.append(metrics)

        df_results = pd.DataFrame(results)

        # Save analysis results
        analysis_file = get_output_filename('tsx_analysis')
        df_results.to_csv(analysis_file, index=False)

        self.results['analysis'] = df_results
        self.results['analysis_file'] = str(analysis_file)

        self.log_complete("Quantitative Analysis", {
            'Stocks Analyzed': len(df_results),
            'Average Sharpe': f"{df_results['sharpe_ratio'].mean():.3f}",
            'Stocks with Sharpe > 1.0': len(df_results[df_results['sharpe_ratio'] > 1.0]),
            'Results File': analysis_file
        })

        return df_results

    def optimize_portfolio(self, analysis_df=None, n_stocks=None, min_sharpe=None):
        """Build optimal portfolio"""
        self.log_start("Portfolio Optimization")

        import pandas as pd

        if analysis_df is None:
            # Load most recent analysis
            analysis_files = list(DIRS['reports'].glob('tsx_analysis_*.csv'))
            if analysis_files:
                latest = max(analysis_files, key=lambda p: p.stat().st_mtime)
                analysis_df = pd.read_csv(latest)
                logger.info(f"Loaded analysis from: {latest.name}")
            else:
                logger.error("No analysis data found. Run --full first.")
                return None

        n_stocks = n_stocks or PORTFOLIO['default_stocks']
        min_sharpe = min_sharpe or ANALYSIS['min_sharpe_threshold']

        # Filter quality stocks
        quality_stocks = analysis_df[
            (analysis_df['sharpe_ratio'] > min_sharpe) &
            (analysis_df['sharpe_ratio'].notna()) &
            (analysis_df['beta'].notna()) &
            (analysis_df['days_of_data'] >= ANALYSIS['min_data_points'])
        ].copy()

        logger.info(f"Filtered to {len(quality_stocks)} quality stocks (Sharpe > {min_sharpe})")

        # Select top stocks by Sharpe
        top_stocks = quality_stocks.nlargest(n_stocks, 'sharpe_ratio')
        tickers = top_stocks['ticker'].tolist()

        logger.info(f"Selected {len(tickers)} stocks for optimization")

        # Optimize
        simulator = PortfolioSimulator(data_dir=str(DIRS['data']))

        max_sharpe = simulator.optimize_portfolio(tickers, objective='sharpe')
        risk_parity = simulator.risk_parity_portfolio(tickers)

        # Save portfolio
        portfolio_data = []
        for ticker in tickers:
            stock_info = analysis_df[analysis_df['ticker'] == ticker].iloc[0]
            portfolio_data.append({
                'ticker': ticker,
                'weight_max_sharpe': max_sharpe['weights'].get(ticker, 0) * 100,
                'weight_risk_parity': risk_parity['weights'].get(ticker, 0) * 100,
                'sharpe_ratio': stock_info['sharpe_ratio'],
                'annual_return': stock_info['annual_return'],
                'annual_volatility': stock_info['annual_volatility'],
                'beta': stock_info['beta'],
                'max_drawdown': stock_info['max_drawdown']
            })

        df_portfolio = pd.DataFrame(portfolio_data)
        df_portfolio = df_portfolio.sort_values('weight_max_sharpe', ascending=False)

        portfolio_file = get_output_filename('optimal_portfolio')
        df_portfolio.to_csv(portfolio_file, index=False)

        self.results['portfolio'] = {
            'max_sharpe': max_sharpe,
            'risk_parity': risk_parity,
            'file': str(portfolio_file)
        }

        self.log_complete("Portfolio Optimization", {
            'Stocks in Portfolio': len(tickers),
            'Expected Return': f"{max_sharpe['return']:.2f}%",
            'Volatility': f"{max_sharpe['volatility']:.2f}%",
            'Sharpe Ratio': f"{max_sharpe['sharpe_ratio']:.3f}",
            'Portfolio File': portfolio_file
        })

        return max_sharpe, risk_parity

    def run_simulation(self, ticker, days=None, simulations=None):
        """Run Monte Carlo simulation"""
        self.log_start(f"Monte Carlo Simulation: {ticker}")

        days = days or SIMULATION['default_days']
        simulations = simulations or SIMULATION['default_simulations']

        sim = AdvancedSimulator(data_dir=str(DIRS['data']))

        # Run multiple simulation methods
        comparison = sim.compare_simulation_methods(ticker, days=days, simulations=simulations)

        # Detailed regime switching
        result = sim.regime_switching_simulation(ticker, days=days, simulations=simulations*2)

        if result:
            logger.info(f"Current Price: ${result['current_price']:.2f}")
            logger.info(f"Forecast ({days} days): ${result['median_final_price']:.2f}")
            logger.info(f"Probability of Profit: {result['probability_profit']:.1f}%")

        self.results['simulation'] = result

        self.log_complete(f"Simulation: {ticker}", {
            'Days Ahead': days,
            'Simulations': simulations,
            'Forecast': f"${result['median_final_price']:.2f}" if result else 'N/A',
            'Probability of Profit': f"{result['probability_profit']:.1f}%" if result else 'N/A'
        })

        return result

    def generate_report(self):
        """Generate comprehensive report"""
        report_file = DIRS['reports'] / f'analysis_report_{self.run_id}.txt'

        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("TSX FINANCIAL ANALYZER - ANALYSIS REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Run ID: {self.run_id}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n")

            # Summary
            f.write("SUMMARY\n")
            f.write("-"*80 + "\n")
            for key, value in self.results.items():
                if isinstance(value, (str, int, float)):
                    f.write(f"{key}: {value}\n")
            f.write("\n")

            # Files generated
            f.write("FILES GENERATED\n")
            f.write("-"*80 + "\n")
            for key, value in self.results.items():
                if isinstance(value, str) and ('file' in key.lower() or value.endswith('.csv')):
                    f.write(f"{key}: {value}\n")

        logger.info(f"Report saved to: {report_file}")
        return report_file

def main():
    parser = argparse.ArgumentParser(
        description='TSX Financial Analyzer - Enterprise Quantitative Analysis System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Main modes
    parser.add_argument('--full', action='store_true', help='Run complete analysis pipeline')
    parser.add_argument('--quick', action='store_true', help='Quick test with limited stocks')
    parser.add_argument('--portfolio', action='store_true', help='Portfolio optimization only')
    parser.add_argument('--simulate', metavar='TICKER', help='Run Monte Carlo simulation for ticker')

    # Options
    parser.add_argument('--stocks', type=int, metavar='N', help='Limit number of stocks to download')
    parser.add_argument('--num-stocks', type=int, metavar='N', help='Number of stocks in portfolio')
    parser.add_argument('--days', type=int, metavar='N', help='Days ahead for simulation')
    parser.add_argument('--sims', type=int, metavar='N', help='Number of simulations')
    parser.add_argument('--min-sharpe', type=float, metavar='X', help='Minimum Sharpe ratio threshold')

    args = parser.parse_args()

    # Create analyzer instance
    analyzer = TSXAnalyzer()

    try:
        if args.full:
            # Complete pipeline
            print("\n" + "="*80)
            print("TSX FINANCIAL ANALYZER - FULL ANALYSIS")
            print("="*80 + "\n")

            analyzer.scrape_tickers()
            analyzer.download_data(limit=args.stocks)
            df = analyzer.analyze_stocks()
            analyzer.optimize_portfolio(df, n_stocks=args.num_stocks, min_sharpe=args.min_sharpe)
            analyzer.generate_report()

        elif args.quick:
            # Quick test
            print("\n" + "="*80)
            print("TSX FINANCIAL ANALYZER - QUICK TEST")
            print("="*80 + "\n")

            limit = args.stocks or 20
            analyzer.scrape_tickers()
            analyzer.download_data(limit=limit)
            df = analyzer.analyze_stocks()
            analyzer.optimize_portfolio(df, n_stocks=min(10, limit))
            analyzer.generate_report()

        elif args.portfolio:
            # Portfolio optimization only
            print("\n" + "="*80)
            print("TSX FINANCIAL ANALYZER - PORTFOLIO OPTIMIZATION")
            print("="*80 + "\n")

            analyzer.optimize_portfolio(n_stocks=args.num_stocks, min_sharpe=args.min_sharpe)
            analyzer.generate_report()

        elif args.simulate:
            # Monte Carlo simulation
            print("\n" + "="*80)
            print(f"TSX FINANCIAL ANALYZER - MONTE CARLO SIMULATION: {args.simulate}")
            print("="*80 + "\n")

            analyzer.run_simulation(args.simulate, days=args.days, simulations=args.sims)
            analyzer.generate_report()

        else:
            parser.print_help()
            return

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print(f"\nRun ID: {analyzer.run_id}")
        print(f"Output Directory: {DIRS['output']}")
        log_file = DIRS['logs'] / f'analysis_{datetime.now().strftime("%Y%m%d")}.log'
        print(f"Log File: {log_file}")

    except Exception as e:
        logger.exception(f"Analysis failed: {e}")
        print(f"\nERROR: {e}")
        print("Check log files for details")
        sys.exit(1)

if __name__ == "__main__":
    main()
