"""
Visualization Module
Creates charts and graphs for analysis results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from simulations import PortfolioSimulator
from quant_analysis import QuantAnalyzer
import os

sns.set_style('darkgrid')

class Visualizer:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.output_dir = 'charts'
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_monte_carlo(self, mc_results, save=True):
        """Plot Monte Carlo simulation results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Monte Carlo Simulation: {mc_results['ticker']}", fontsize=16, fontweight='bold')

        # Plot 1: Sample price paths
        ax1 = axes[0, 0]
        paths = mc_results['simulation_paths']
        n_paths_to_plot = min(100, mc_results['simulations'])
        for i in range(n_paths_to_plot):
            ax1.plot(paths[:, i], alpha=0.1, color='blue')
        ax1.axhline(y=mc_results['current_price'], color='red', linestyle='--', label='Current Price')
        ax1.set_title('Sample Price Paths')
        ax1.set_xlabel('Days')
        ax1.set_ylabel('Price ($)')
        ax1.legend()

        # Plot 2: Final price distribution
        ax2 = axes[0, 1]
        final_prices = paths[-1, :]
        ax2.hist(final_prices, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(mc_results['current_price'], color='red', linestyle='--', label='Current Price')
        ax2.axvline(mc_results['mean_final_price'], color='blue', linestyle='--', label='Mean Prediction')
        ax2.set_title('Distribution of Final Prices')
        ax2.set_xlabel('Price ($)')
        ax2.set_ylabel('Frequency')
        ax2.legend()

        # Plot 3: Percentile ranges
        ax3 = axes[1, 0]
        percentiles = [5, 25, 50, 75, 95]
        values = [mc_results[f'percentile_{p}'] if p != 50 else mc_results['median_final_price'] for p in percentiles]
        ax3.barh(range(len(percentiles)), values, color=['red', 'orange', 'yellow', 'lightgreen', 'green'])
        ax3.set_yticks(range(len(percentiles)))
        ax3.set_yticklabels([f'{p}th' for p in percentiles])
        ax3.axvline(mc_results['current_price'], color='black', linestyle='--', linewidth=2, label='Current Price')
        ax3.set_title('Price Percentiles')
        ax3.set_xlabel('Price ($)')
        ax3.legend()

        # Plot 4: Statistics summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        stats_text = f"""
        Current Price: ${mc_results['current_price']:.2f}

        Predicted Price ({mc_results['days_ahead']} days):
        Mean: ${mc_results['mean_final_price']:.2f}
        Median: ${mc_results['median_final_price']:.2f}

        Range:
        Min: ${mc_results['min_final_price']:.2f}
        Max: ${mc_results['max_final_price']:.2f}

        Confidence Intervals:
        5%-95%: ${mc_results['percentile_5']:.2f} - ${mc_results['percentile_95']:.2f}
        25%-75%: ${mc_results['percentile_25']:.2f} - ${mc_results['percentile_75']:.2f}

        Probability of Profit: {mc_results['probability_profit']:.1f}%

        Simulations: {mc_results['simulations']:,}
        """
        ax4.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center', fontfamily='monospace')
        ax4.set_title('Summary Statistics')

        plt.tight_layout()

        if save:
            filename = os.path.join(self.output_dir, f"monte_carlo_{mc_results['ticker'].replace('.', '_')}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")

        plt.show()

    def plot_efficient_frontier(self, tickers, save=True):
        """Plot efficient frontier"""
        simulator = PortfolioSimulator(self.data_dir)
        ef_data = simulator.efficient_frontier(tickers, n_portfolios=1000)

        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot random portfolios
        portfolios = ef_data['random_portfolios']
        returns = [p['return'] for p in portfolios]
        vols = [p['volatility'] for p in portfolios]
        sharpes = [p['sharpe'] for p in portfolios]

        scatter = ax.scatter(vols, returns, c=sharpes, cmap='viridis', alpha=0.5, s=20)
        plt.colorbar(scatter, ax=ax, label='Sharpe Ratio')

        # Plot max Sharpe portfolio
        max_sharpe = ef_data['max_sharpe_portfolio']
        ax.scatter(max_sharpe['volatility'], max_sharpe['return'],
                  color='red', marker='*', s=500, label='Max Sharpe Ratio', edgecolors='black', linewidth=2)

        # Plot min volatility portfolio
        min_vol = ef_data['min_volatility_portfolio']
        ax.scatter(min_vol['volatility'], min_vol['return'],
                  color='blue', marker='*', s=500, label='Min Volatility', edgecolors='black', linewidth=2)

        ax.set_xlabel('Volatility (%)', fontsize=12)
        ax.set_ylabel('Return (%)', fontsize=12)
        ax.set_title(f'Efficient Frontier\nTickers: {", ".join(tickers)}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filename = os.path.join(self.output_dir, 'efficient_frontier.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")

        plt.show()

    def plot_portfolio_comparison(self, tickers, save=True):
        """Compare different portfolio strategies"""
        simulator = PortfolioSimulator(self.data_dir)

        max_sharpe = simulator.optimize_portfolio(tickers, 'sharpe')
        min_vol = simulator.optimize_portfolio(tickers, 'min_vol')
        risk_parity = simulator.risk_parity_portfolio(tickers)
        equal_weight = {ticker: 1/len(tickers) for ticker in tickers}

        portfolios = {
            'Max Sharpe': max_sharpe,
            'Min Volatility': min_vol,
            'Risk Parity': risk_parity,
            'Equal Weight': {
                'weights': equal_weight,
                'return': simulator.portfolio_return(np.array(list(equal_weight.values())),
                                                    simulator.load_multiple_stocks(tickers).pct_change().dropna()) * 100,
                'volatility': simulator.portfolio_volatility(np.array(list(equal_weight.values())),
                                                            simulator.load_multiple_stocks(tickers).pct_change().dropna()) * 100,
                'sharpe_ratio': 0
            }
        }

        # Recalculate Sharpe for equal weight
        ew_ret = portfolios['Equal Weight']['return']
        ew_vol = portfolios['Equal Weight']['volatility']
        portfolios['Equal Weight']['sharpe_ratio'] = (ew_ret - 4.5) / ew_vol

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Portfolio Strategy Comparison\n{", ".join(tickers)}', fontsize=16, fontweight='bold')

        # Plot 1: Returns comparison
        ax1 = axes[0, 0]
        returns = [p['return'] for p in portfolios.values()]
        ax1.bar(portfolios.keys(), returns, color=['red', 'blue', 'green', 'orange'])
        ax1.set_ylabel('Annual Return (%)')
        ax1.set_title('Expected Returns')
        ax1.tick_params(axis='x', rotation=45)

        # Plot 2: Volatility comparison
        ax2 = axes[0, 1]
        vols = [p['volatility'] for p in portfolios.values()]
        ax2.bar(portfolios.keys(), vols, color=['red', 'blue', 'green', 'orange'])
        ax2.set_ylabel('Annual Volatility (%)')
        ax2.set_title('Portfolio Volatility')
        ax2.tick_params(axis='x', rotation=45)

        # Plot 3: Sharpe Ratio comparison
        ax3 = axes[1, 0]
        sharpes = [p['sharpe_ratio'] for p in portfolios.values()]
        ax3.bar(portfolios.keys(), sharpes, color=['red', 'blue', 'green', 'orange'])
        ax3.set_ylabel('Sharpe Ratio')
        ax3.set_title('Risk-Adjusted Returns (Sharpe Ratio)')
        ax3.tick_params(axis='x', rotation=45)

        # Plot 4: Return vs Volatility
        ax4 = axes[1, 1]
        for name, portfolio in portfolios.items():
            ax4.scatter(portfolio['volatility'], portfolio['return'], s=200, label=name)
        ax4.set_xlabel('Volatility (%)')
        ax4.set_ylabel('Return (%)')
        ax4.set_title('Risk-Return Profile')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filename = os.path.join(self.output_dir, 'portfolio_comparison.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")

        plt.show()

    def plot_correlation_matrix(self, tickers, save=True):
        """Plot correlation matrix of returns"""
        simulator = PortfolioSimulator(self.data_dir)
        prices = simulator.load_multiple_stocks(tickers)
        returns = prices.pct_change().dropna()

        # Calculate correlation matrix
        corr_matrix = returns.corr()

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})

        ax.set_title('Return Correlation Matrix', fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()

        if save:
            filename = os.path.join(self.output_dir, 'correlation_matrix.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")

        plt.show()

    def plot_top_performers(self, analysis_csv='tsx_analysis.csv', save=True):
        """Plot top performing stocks"""
        df = pd.read_csv(analysis_csv)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('TSX Top Performers Analysis', fontsize=16, fontweight='bold')

        # Top by Sharpe Ratio
        ax1 = axes[0, 0]
        top_sharpe = df.nlargest(10, 'sharpe_ratio')[['ticker', 'sharpe_ratio']]
        ax1.barh(top_sharpe['ticker'], top_sharpe['sharpe_ratio'], color='green')
        ax1.set_xlabel('Sharpe Ratio')
        ax1.set_title('Top 10 by Sharpe Ratio')
        ax1.invert_yaxis()

        # Top by Return
        ax2 = axes[0, 1]
        top_return = df.nlargest(10, 'annual_return')[['ticker', 'annual_return']]
        ax2.barh(top_return['ticker'], top_return['annual_return'], color='blue')
        ax2.set_xlabel('Annual Return (%)')
        ax2.set_title('Top 10 by Annual Return')
        ax2.invert_yaxis()

        # Lowest Volatility
        ax3 = axes[1, 0]
        low_vol = df.nsmallest(10, 'annual_volatility')[['ticker', 'annual_volatility']]
        ax3.barh(low_vol['ticker'], low_vol['annual_volatility'], color='purple')
        ax3.set_xlabel('Annual Volatility (%)')
        ax3.set_title('Top 10 Lowest Volatility')
        ax3.invert_yaxis()

        # Risk-Return Scatter
        ax4 = axes[1, 1]
        ax4.scatter(df['annual_volatility'], df['annual_return'], alpha=0.5)
        ax4.set_xlabel('Annual Volatility (%)')
        ax4.set_ylabel('Annual Return (%)')
        ax4.set_title('Risk-Return Profile (All TSX Stocks)')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filename = os.path.join(self.output_dir, 'top_performers.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")

        plt.show()

if __name__ == "__main__":
    viz = Visualizer(data_dir='data')

    # Example: Monte Carlo visualization
    print("Generating Monte Carlo visualization...")
    simulator = PortfolioSimulator(data_dir='data')
    mc_results = simulator.monte_carlo_simulation('RY.TO', days=252, simulations=5000)
    if mc_results:
        viz.plot_monte_carlo(mc_results)

    # Example: Efficient Frontier
    print("\nGenerating Efficient Frontier...")
    tickers = ['RY.TO', 'TD.TO', 'BNS.TO', 'BMO.TO', 'CM.TO']
    viz.plot_efficient_frontier(tickers)

    # Example: Portfolio Comparison
    print("\nGenerating Portfolio Comparison...")
    viz.plot_portfolio_comparison(tickers)

    # Example: Correlation Matrix
    print("\nGenerating Correlation Matrix...")
    viz.plot_correlation_matrix(tickers)

    # Example: Top Performers (if analysis has been run)
    if os.path.exists('tsx_analysis.csv'):
        print("\nGenerating Top Performers visualization...")
        viz.plot_top_performers()

    print(f"\nAll charts saved to: {os.path.abspath('charts')}")
