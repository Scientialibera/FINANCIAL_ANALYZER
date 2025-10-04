"""
Monte Carlo Simulations and Portfolio Optimization
Includes: Monte Carlo price simulations, portfolio optimization,
efficient frontier, risk parity, and scenario analysis
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

class PortfolioSimulator:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir

    def load_stock_data(self, ticker):
        """Load historical data for a stock"""
        csv_path = os.path.join(self.data_dir, f"{ticker.replace('.', '_')}_history.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            return df['Close']
        return None

    def load_multiple_stocks(self, tickers):
        """Load data for multiple stocks into a DataFrame"""
        data = {}
        for ticker in tickers:
            prices = self.load_stock_data(ticker)
            if prices is not None:
                data[ticker] = prices

        df = pd.DataFrame(data)
        return df.dropna()

    def monte_carlo_simulation(self, ticker, days=252, simulations=10000):
        """
        Monte Carlo simulation for stock price prediction
        Uses Geometric Brownian Motion

        S(t+1) = S(t) * exp((μ - σ²/2)Δt + σ√Δt * Z)
        where Z ~ N(0,1)
        """
        # Load historical data
        prices = self.load_stock_data(ticker)
        if prices is None:
            print(f"No data for {ticker}")
            return None

        # Calculate historical parameters
        returns = prices.pct_change().dropna()
        mu = returns.mean()  # drift
        sigma = returns.std()  # volatility

        # Current price
        S0 = prices.iloc[-1]

        # Time parameters
        dt = 1  # daily steps

        # Initialize simulation matrix
        simulation_results = np.zeros((days, simulations))

        # Run simulations
        for sim in range(simulations):
            prices_sim = np.zeros(days)
            prices_sim[0] = S0

            for t in range(1, days):
                # Generate random shock
                Z = np.random.standard_normal()

                # Calculate next price using GBM
                prices_sim[t] = prices_sim[t-1] * np.exp(
                    (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
                )

            simulation_results[:, sim] = prices_sim

        # Calculate statistics
        final_prices = simulation_results[-1, :]

        results = {
            'ticker': ticker,
            'current_price': S0,
            'simulations': simulations,
            'days_ahead': days,
            'mean_final_price': final_prices.mean(),
            'median_final_price': np.median(final_prices),
            'std_final_price': final_prices.std(),
            'min_final_price': final_prices.min(),
            'max_final_price': final_prices.max(),
            'percentile_5': np.percentile(final_prices, 5),
            'percentile_25': np.percentile(final_prices, 25),
            'percentile_75': np.percentile(final_prices, 75),
            'percentile_95': np.percentile(final_prices, 95),
            'probability_profit': (final_prices > S0).sum() / simulations * 100,
            'simulation_paths': simulation_results
        }

        return results

    def portfolio_return(self, weights, returns):
        """Calculate portfolio return"""
        return np.sum(returns.mean() * weights) * 252

    def portfolio_volatility(self, weights, returns):
        """Calculate portfolio volatility"""
        return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))

    def portfolio_sharpe_ratio(self, weights, returns, risk_free_rate=0.045):
        """Calculate portfolio Sharpe ratio"""
        p_ret = self.portfolio_return(weights, returns)
        p_vol = self.portfolio_volatility(weights, returns)
        return (p_ret - risk_free_rate) / p_vol

    def negative_sharpe(self, weights, returns, risk_free_rate=0.045):
        """Negative Sharpe ratio for minimization"""
        return -self.portfolio_sharpe_ratio(weights, returns, risk_free_rate)

    def optimize_portfolio(self, tickers, objective='sharpe', target_return=None):
        """
        Optimize portfolio weights

        objective: 'sharpe' (maximize Sharpe), 'min_vol' (minimize volatility),
                   'max_return' (maximize return)
        """
        # Load data
        prices = self.load_multiple_stocks(tickers)
        returns = prices.pct_change().dropna()

        n_assets = len(tickers)

        # Constraints: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

        # Bounds: 0 <= weight <= 1 (no shorting)
        bounds = tuple((0, 1) for _ in range(n_assets))

        # Initial guess: equal weights
        init_weights = np.array([1/n_assets] * n_assets)

        if objective == 'sharpe':
            # Maximize Sharpe ratio
            result = minimize(
                self.negative_sharpe,
                init_weights,
                args=(returns,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
        elif objective == 'min_vol':
            # Minimize volatility
            result = minimize(
                self.portfolio_volatility,
                init_weights,
                args=(returns,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
        elif objective == 'max_return':
            # Maximize return
            result = minimize(
                lambda w, r: -self.portfolio_return(w, r),
                init_weights,
                args=(returns,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )

        # Calculate portfolio metrics
        optimal_weights = result.x
        port_return = self.portfolio_return(optimal_weights, returns)
        port_vol = self.portfolio_volatility(optimal_weights, returns)
        port_sharpe = self.portfolio_sharpe_ratio(optimal_weights, returns)

        return {
            'weights': dict(zip(tickers, optimal_weights)),
            'return': port_return * 100,
            'volatility': port_vol * 100,
            'sharpe_ratio': port_sharpe
        }

    def efficient_frontier(self, tickers, n_portfolios=100):
        """
        Generate efficient frontier

        Returns portfolios with different risk-return profiles
        """
        prices = self.load_multiple_stocks(tickers)
        returns = prices.pct_change().dropna()
        n_assets = len(tickers)

        results = []

        # Generate random portfolios
        for _ in range(n_portfolios):
            weights = np.random.random(n_assets)
            weights /= np.sum(weights)

            port_return = self.portfolio_return(weights, returns)
            port_vol = self.portfolio_volatility(weights, returns)
            port_sharpe = self.portfolio_sharpe_ratio(weights, returns)

            results.append({
                'return': port_return * 100,
                'volatility': port_vol * 100,
                'sharpe': port_sharpe,
                'weights': weights
            })

        # Add optimized portfolios
        max_sharpe = self.optimize_portfolio(tickers, 'sharpe')
        min_vol = self.optimize_portfolio(tickers, 'min_vol')

        return {
            'random_portfolios': results,
            'max_sharpe_portfolio': max_sharpe,
            'min_volatility_portfolio': min_vol
        }

    def risk_parity_portfolio(self, tickers):
        """
        Risk Parity Portfolio
        Each asset contributes equally to portfolio risk
        """
        prices = self.load_multiple_stocks(tickers)
        returns = prices.pct_change().dropna()
        n_assets = len(tickers)

        # Objective: minimize difference in risk contributions
        def risk_contribution_diff(weights):
            port_vol = self.portfolio_volatility(weights, returns)
            cov_matrix = returns.cov() * 252
            marginal_contrib = np.dot(cov_matrix, weights)
            contrib = weights * marginal_contrib / port_vol
            target_contrib = port_vol / n_assets
            return np.sum((contrib - target_contrib) ** 2)

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        init_weights = np.array([1/n_assets] * n_assets)

        result = minimize(
            risk_contribution_diff,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        optimal_weights = result.x
        port_return = self.portfolio_return(optimal_weights, returns)
        port_vol = self.portfolio_volatility(optimal_weights, returns)
        port_sharpe = self.portfolio_sharpe_ratio(optimal_weights, returns)

        return {
            'weights': dict(zip(tickers, optimal_weights)),
            'return': port_return * 100,
            'volatility': port_vol * 100,
            'sharpe_ratio': port_sharpe
        }

    def backtest_portfolio(self, weights_dict, start_date=None, end_date=None):
        """
        Backtest a portfolio with given weights
        """
        tickers = list(weights_dict.keys())
        weights = np.array(list(weights_dict.values()))

        # Load data
        prices = self.load_multiple_stocks(tickers)

        # Filter by date if specified
        if start_date:
            prices = prices[prices.index >= start_date]
        if end_date:
            prices = prices[prices.index <= end_date]

        # Calculate portfolio value
        returns = prices.pct_change()
        portfolio_returns = (returns * weights).sum(axis=1)
        portfolio_value = (1 + portfolio_returns).cumprod()

        # Calculate metrics
        total_return = (portfolio_value.iloc[-1] - 1) * 100
        annual_return = portfolio_returns.mean() * 252 * 100
        annual_vol = portfolio_returns.std() * np.sqrt(252) * 100
        sharpe = (annual_return - 4.5) / annual_vol if annual_vol > 0 else 0

        # Maximum Drawdown
        cummax = portfolio_value.cummax()
        drawdown = (portfolio_value - cummax) / cummax
        max_drawdown = drawdown.min() * 100

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'portfolio_value': portfolio_value,
            'daily_returns': portfolio_returns
        }

if __name__ == "__main__":
    sim = PortfolioSimulator(data_dir='data')

    # Example: Monte Carlo simulation for Royal Bank
    print("Running Monte Carlo simulation for RY.TO...")
    mc_results = sim.monte_carlo_simulation('RY.TO', days=252, simulations=10000)

    if mc_results:
        print(f"\nMonte Carlo Results for {mc_results['ticker']} (1 year ahead):")
        print(f"Current Price: ${mc_results['current_price']:.2f}")
        print(f"Mean Predicted Price: ${mc_results['mean_final_price']:.2f}")
        print(f"Median Predicted Price: ${mc_results['median_final_price']:.2f}")
        print(f"5th Percentile: ${mc_results['percentile_5']:.2f}")
        print(f"95th Percentile: ${mc_results['percentile_95']:.2f}")
        print(f"Probability of Profit: {mc_results['probability_profit']:.1f}%")

    # Example: Portfolio optimization
    print("\n" + "="*80)
    print("Portfolio Optimization Example:")
    bank_stocks = ['RY.TO', 'TD.TO', 'BNS.TO', 'BMO.TO', 'CM.TO']

    print(f"\nOptimizing portfolio of: {', '.join(bank_stocks)}")

    # Maximum Sharpe Ratio
    max_sharpe = sim.optimize_portfolio(bank_stocks, objective='sharpe')
    print("\nMaximum Sharpe Ratio Portfolio:")
    print(f"Expected Return: {max_sharpe['return']:.2f}%")
    print(f"Volatility: {max_sharpe['volatility']:.2f}%")
    print(f"Sharpe Ratio: {max_sharpe['sharpe_ratio']:.3f}")
    print("Weights:")
    for ticker, weight in max_sharpe['weights'].items():
        print(f"  {ticker}: {weight*100:.2f}%")

    # Minimum Volatility
    min_vol = sim.optimize_portfolio(bank_stocks, objective='min_vol')
    print("\nMinimum Volatility Portfolio:")
    print(f"Expected Return: {min_vol['return']:.2f}%")
    print(f"Volatility: {min_vol['volatility']:.2f}%")
    print(f"Sharpe Ratio: {min_vol['sharpe_ratio']:.3f}")
    print("Weights:")
    for ticker, weight in min_vol['weights'].items():
        print(f"  {ticker}: {weight*100:.2f}%")

    # Risk Parity
    risk_parity = sim.risk_parity_portfolio(bank_stocks)
    print("\nRisk Parity Portfolio:")
    print(f"Expected Return: {risk_parity['return']:.2f}%")
    print(f"Volatility: {risk_parity['volatility']:.2f}%")
    print(f"Sharpe Ratio: {risk_parity['sharpe_ratio']:.3f}")
    print("Weights:")
    for ticker, weight in risk_parity['weights'].items():
        print(f"  {ticker}: {weight*100:.2f}%")
