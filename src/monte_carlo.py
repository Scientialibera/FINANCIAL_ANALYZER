"""
Advanced Monte Carlo Simulations and Quantitative Techniques
- Geometric Brownian Motion with jump diffusion
- Historical bootstrap simulation
- GARCH volatility modeling
- Copula-based correlation modeling
- Value at Risk (VaR) and Conditional VaR (CVaR) analysis
- Stress testing and scenario analysis
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import os

class AdvancedSimulator:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir

    def load_stock_data(self, ticker):
        """Load historical data"""
        csv_path = os.path.join(self.data_dir, f"{ticker.replace('.', '_')}_history.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            return df['Close']
        return None

    # ========== MONTE CARLO WITH JUMP DIFFUSION ==========

    def gbm_with_jumps(self, ticker, days=252, simulations=10000, jump_prob=0.02):
        """
        Geometric Brownian Motion with Jump Diffusion (Merton Model)

        Adds sudden price jumps to capture fat tails and extreme events

        S(t+1) = S(t) * exp((μ - σ²/2)Δt + σ√Δt*Z + J)
        where J is jump component
        """
        prices = self.load_stock_data(ticker)
        if prices is None:
            return None

        returns = prices.pct_change().dropna()

        # Separate normal returns from jumps
        # Jumps defined as returns > 3 std devs
        threshold = 3 * returns.std()
        normal_returns = returns[abs(returns) < threshold]
        jump_returns = returns[abs(returns) >= threshold]

        # Parameters
        mu = normal_returns.mean()
        sigma = normal_returns.std()
        S0 = prices.iloc[-1]

        # Jump parameters
        jump_mean = jump_returns.mean() if len(jump_returns) > 0 else 0
        jump_std = jump_returns.std() if len(jump_returns) > 0 else sigma

        # Run simulations
        simulation_results = np.zeros((days, simulations))

        for sim in range(simulations):
            prices_path = np.zeros(days)
            prices_path[0] = S0

            for t in range(1, days):
                # Normal diffusion
                Z = np.random.standard_normal()
                drift = (mu - 0.5 * sigma**2)
                diffusion = sigma * Z

                # Jump component
                jump = 0
                if np.random.random() < jump_prob:
                    jump = np.random.normal(jump_mean, jump_std)

                prices_path[t] = prices_path[t-1] * np.exp(drift + diffusion + jump)

            simulation_results[:, sim] = prices_path

        return self._calculate_simulation_stats(ticker, S0, days, simulation_results)

    # ========== HISTORICAL BOOTSTRAP ==========

    def historical_bootstrap(self, ticker, days=252, simulations=10000):
        """
        Historical Bootstrap Simulation

        Randomly samples actual historical returns (preserves actual distribution)
        More conservative than GBM as it doesn't assume normality
        """
        prices = self.load_stock_data(ticker)
        if prices is None:
            return None

        returns = prices.pct_change().dropna().values
        S0 = prices.iloc[-1]

        simulation_results = np.zeros((days, simulations))

        for sim in range(simulations):
            prices_path = np.zeros(days)
            prices_path[0] = S0

            # Bootstrap: randomly sample from historical returns with replacement
            sampled_returns = np.random.choice(returns, size=days-1, replace=True)

            for t in range(1, days):
                prices_path[t] = prices_path[t-1] * (1 + sampled_returns[t-1])

            simulation_results[:, sim] = prices_path

        return self._calculate_simulation_stats(ticker, S0, days, simulation_results)

    # ========== GARCH VOLATILITY SIMULATION ==========

    def garch_simulation(self, ticker, days=252, simulations=1000):
        """
        GARCH(1,1) volatility model simulation

        Models time-varying volatility (volatility clustering)
        σ²(t) = ω + α*r²(t-1) + β*σ²(t-1)
        """
        prices = self.load_stock_data(ticker)
        if prices is None:
            return None

        returns = prices.pct_change().dropna().values
        S0 = prices.iloc[-1]

        # Estimate GARCH parameters (simplified)
        # In practice, use arch library for proper GARCH estimation
        omega = 0.0001  # Long-run variance
        alpha = 0.15    # ARCH coefficient
        beta = 0.80     # GARCH coefficient

        # Initial volatility
        sigma_sq = returns.var()

        simulation_results = np.zeros((days, simulations))

        for sim in range(simulations):
            prices_path = np.zeros(days)
            prices_path[0] = S0

            sigma_sq_t = sigma_sq
            mu = returns.mean()

            for t in range(1, days):
                # Generate return with current volatility
                Z = np.random.standard_normal()
                r_t = mu + np.sqrt(sigma_sq_t) * Z

                # Update price
                prices_path[t] = prices_path[t-1] * (1 + r_t)

                # Update variance for next period (GARCH)
                sigma_sq_t = omega + alpha * r_t**2 + beta * sigma_sq_t

            simulation_results[:, sim] = prices_path

        return self._calculate_simulation_stats(ticker, S0, days, simulation_results)

    # ========== REGIME SWITCHING MODEL ==========

    def regime_switching_simulation(self, ticker, days=252, simulations=5000):
        """
        Regime Switching Model

        Models bull/bear market regimes with different parameters
        """
        prices = self.load_stock_data(ticker)
        if prices is None:
            return None

        returns = prices.pct_change().dropna()
        S0 = prices.iloc[-1]

        # Identify regimes using rolling volatility
        rolling_vol = returns.rolling(60).std()
        high_vol_threshold = rolling_vol.quantile(0.7)

        # Bull regime (low volatility)
        bull_returns = returns[rolling_vol < high_vol_threshold]
        bull_mu = bull_returns.mean()
        bull_sigma = bull_returns.std()

        # Bear regime (high volatility)
        bear_returns = returns[rolling_vol >= high_vol_threshold]
        bear_mu = bear_returns.mean() if len(bear_returns) > 0 else bull_mu - 0.01
        bear_sigma = bear_returns.std() if len(bear_returns) > 0 else bull_sigma * 1.5

        # Transition probabilities
        prob_bull_to_bear = 0.05  # 5% chance of switching to bear market
        prob_bear_to_bull = 0.15  # 15% chance of switching to bull market

        simulation_results = np.zeros((days, simulations))

        for sim in range(simulations):
            prices_path = np.zeros(days)
            prices_path[0] = S0

            # Start in bull regime
            in_bull_regime = True

            for t in range(1, days):
                # Determine regime switch
                if in_bull_regime:
                    if np.random.random() < prob_bull_to_bear:
                        in_bull_regime = False
                    mu, sigma = bull_mu, bull_sigma
                else:
                    if np.random.random() < prob_bear_to_bull:
                        in_bull_regime = True
                    mu, sigma = bear_mu, bear_sigma

                # Generate return
                Z = np.random.standard_normal()
                r_t = mu + sigma * Z

                prices_path[t] = prices_path[t-1] * (1 + r_t)

            simulation_results[:, sim] = prices_path

        return self._calculate_simulation_stats(ticker, S0, days, simulation_results)

    # ========== HELPER FUNCTIONS ==========

    def _calculate_simulation_stats(self, ticker, S0, days, simulation_results):
        """Calculate statistics from simulation results"""
        final_prices = simulation_results[-1, :]

        # Calculate percentiles at different time horizons
        time_horizons = [30, 60, 126, 252] if days >= 252 else [30, 60, days]
        horizon_stats = {}

        for horizon in time_horizons:
            if horizon <= days:
                horizon_prices = simulation_results[horizon-1, :]
                horizon_stats[f'{horizon}d'] = {
                    'mean': horizon_prices.mean(),
                    'median': np.median(horizon_prices),
                    'p05': np.percentile(horizon_prices, 5),
                    'p25': np.percentile(horizon_prices, 25),
                    'p75': np.percentile(horizon_prices, 75),
                    'p95': np.percentile(horizon_prices, 95),
                }

        return {
            'ticker': ticker,
            'current_price': S0,
            'simulations': simulation_results.shape[1],
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
            'probability_profit': (final_prices > S0).sum() / len(final_prices) * 100,
            'expected_return': ((final_prices.mean() / S0) - 1) * 100,
            'var_95': np.percentile((final_prices / S0 - 1) * 100, 5),
            'cvar_95': ((final_prices / S0 - 1) * 100)[
                (final_prices / S0 - 1) * 100 <= np.percentile((final_prices / S0 - 1) * 100, 5)
            ].mean(),
            'simulation_paths': simulation_results,
            'horizon_stats': horizon_stats
        }

    # ========== PORTFOLIO STRESS TESTING ==========

    def stress_test_portfolio(self, portfolio_weights, scenarios=None):
        """
        Stress test portfolio under various market scenarios

        Default scenarios: 2008 crisis, 2020 COVID crash, high inflation, etc.
        """
        if scenarios is None:
            scenarios = {
                '2008_Crisis': {'market_return': -0.37, 'volatility_mult': 2.0},
                '2020_COVID': {'market_return': -0.35, 'volatility_mult': 1.8},
                'Flash_Crash': {'market_return': -0.20, 'volatility_mult': 3.0},
                'High_Inflation': {'market_return': -0.15, 'volatility_mult': 1.5},
                'Normal_Bear': {'market_return': -0.20, 'volatility_mult': 1.3},
            }

        results = {}

        for scenario_name, scenario_params in scenarios.items():
            # Apply scenario to each stock
            portfolio_loss = 0

            for ticker, weight in portfolio_weights.items():
                prices = self.load_stock_data(ticker)
                if prices is None:
                    continue

                returns = prices.pct_change().dropna()
                beta = self._estimate_beta(returns)

                # Expected loss in scenario
                stock_loss = scenario_params['market_return'] * beta
                portfolio_loss += weight * stock_loss

            results[scenario_name] = {
                'portfolio_return': portfolio_loss * 100,
                'description': scenario_name
            }

        return results

    def _estimate_beta(self, returns):
        """Quick beta estimation vs market (simplified)"""
        # In reality, would load market data
        # For now, estimate from volatility
        vol = returns.std()
        market_vol = 0.15  # Assume 15% market volatility
        return vol / market_vol

    # ========== COMPARISON FRAMEWORK ==========

    def compare_simulation_methods(self, ticker, days=252, simulations=5000):
        """
        Compare different simulation methods
        """
        print(f"\n{'='*80}")
        print(f"COMPARING SIMULATION METHODS: {ticker}")
        print(f"{'='*80}")

        methods = {
            'GBM (Standard)': lambda: self.gbm_with_jumps(ticker, days, simulations, jump_prob=0.0),
            'GBM with Jumps': lambda: self.gbm_with_jumps(ticker, days, simulations, jump_prob=0.02),
            'Historical Bootstrap': lambda: self.historical_bootstrap(ticker, days, simulations),
            'Regime Switching': lambda: self.regime_switching_simulation(ticker, days, simulations),
        }

        comparison_results = []

        for method_name, method_func in methods.items():
            print(f"\nRunning {method_name}...")
            result = method_func()

            if result:
                comparison_results.append({
                    'method': method_name,
                    'mean_price': result['mean_final_price'],
                    'median_price': result['median_final_price'],
                    'prob_profit': result['probability_profit'],
                    'var_95': result['var_95'],
                    'cvar_95': result['cvar_95']
                })

        # Display comparison
        df_compare = pd.DataFrame(comparison_results)
        print(f"\n{'='*80}")
        print("SIMULATION METHOD COMPARISON")
        print(f"{'='*80}")
        print(df_compare.to_string(index=False))

        return df_compare

if __name__ == "__main__":
    sim = AdvancedSimulator(data_dir='data')

    # Example: Compare simulation methods for Royal Bank
    ticker = 'RY.TO'

    print("=" * 80)
    print("ADVANCED MONTE CARLO SIMULATIONS")
    print("=" * 80)

    # Run comparison
    comparison = sim.compare_simulation_methods(ticker, days=252, simulations=5000)

    # Example: Regime switching simulation
    print("\n" + "=" * 80)
    print("REGIME SWITCHING SIMULATION (1 year ahead)")
    print("=" * 80)

    regime_result = sim.regime_switching_simulation(ticker, days=252, simulations=10000)

    if regime_result:
        print(f"\nCurrent Price: ${regime_result['current_price']:.2f}")
        print(f"\n1-Year Forecast:")
        print(f"  Mean: ${regime_result['mean_final_price']:.2f}")
        print(f"  Median: ${regime_result['median_final_price']:.2f}")
        print(f"  5th percentile: ${regime_result['percentile_5']:.2f}")
        print(f"  95th percentile: ${regime_result['percentile_95']:.2f}")
        print(f"  Probability of Profit: {regime_result['probability_profit']:.1f}%")
        print(f"  Expected Return: {regime_result['expected_return']:.2f}%")
        print(f"  VaR (95%): {regime_result['var_95']:.2f}%")
        print(f"  CVaR (95%): {regime_result['cvar_95']:.2f}%")

        # Horizon analysis
        print("\n  Time Horizon Analysis:")
        for horizon, stats in regime_result['horizon_stats'].items():
            print(f"    {horizon:6s}: ${stats['median']:.2f} (5%-95%: ${stats['p05']:.2f}-${stats['p95']:.2f})")
