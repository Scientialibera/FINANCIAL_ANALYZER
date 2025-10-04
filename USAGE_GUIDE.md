# TSX Stock Analysis - Usage Guide

## Quick Start (5 minutes)

### 1. Test the System
```bash
python quick_test.py
```
This will:
- Download data for 5 major Canadian banks
- Run quantitative analysis
- Perform Monte Carlo simulation
- Optimize portfolios

### 2. Run Full Analysis
```bash
python main.py
```
Follow the interactive prompts to:
1. Download all TSX stock data (~200+ companies)
2. Analyze all stocks (generates `tsx_analysis.csv`)
3. Run Monte Carlo simulations
4. Optimize portfolios

## Individual Module Usage

### Download TSX Data
```bash
python tsx_data_downloader.py
```
Downloads ALL TSX stock data to `data/` folder.

### Analyze All Stocks
```bash
python quant_analysis.py
```
Generates `tsx_analysis.csv` with metrics for all stocks.

### Generate Visualizations
```bash
python visualizations.py
```
Creates charts in `charts/` folder.

## Python API Examples

### Example 1: Analyze a Specific Stock
```python
from quant_analysis import QuantAnalyzer
import yfinance as yf

# Setup
analyzer = QuantAnalyzer(data_dir='data')
market = yf.Ticker('^GSPTSE').history(period='max')

# Analyze Royal Bank
metrics = analyzer.analyze_stock('RY.TO', market)

print(f"Annual Return: {metrics['annual_return']:.2f}%")
print(f"Volatility: {metrics['annual_volatility']:.2f}%")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
print(f"Beta: {metrics['beta']:.3f}")
print(f"Alpha: {metrics['alpha']:.2f}%")
print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
```

### Example 2: Monte Carlo Price Prediction
```python
from simulations import PortfolioSimulator

sim = PortfolioSimulator(data_dir='data')

# Predict Shopify price 1 year ahead
mc = sim.monte_carlo_simulation('SHOP.TO', days=252, simulations=10000)

print(f"Current Price: ${mc['current_price']:.2f}")
print(f"\n1-Year Forecast:")
print(f"  Mean: ${mc['mean_final_price']:.2f}")
print(f"  Median: ${mc['median_final_price']:.2f}")
print(f"  5th percentile: ${mc['percentile_5']:.2f}")
print(f"  95th percentile: ${mc['percentile_95']:.2f}")
print(f"\nProbability of profit: {mc['probability_profit']:.1f}%")
```

### Example 3: Find Best Portfolio
```python
from simulations import PortfolioSimulator

sim = PortfolioSimulator(data_dir='data')

# Tech stocks
tech = ['SHOP.TO', 'BB.TO', 'LSPD.TO', 'DCBO.TO', 'TOI.TO']

# Maximum Sharpe Ratio
optimal = sim.optimize_portfolio(tech, objective='sharpe')

print("Optimal Portfolio:")
print(f"Expected Return: {optimal['return']:.2f}%")
print(f"Volatility: {optimal['volatility']:.2f}%")
print(f"Sharpe Ratio: {optimal['sharpe_ratio']:.3f}")
print("\nWeights:")
for ticker, weight in sorted(optimal['weights'].items(), key=lambda x: x[1], reverse=True):
    if weight > 0.01:
        print(f"  {ticker}: {weight*100:.1f}%")
```

### Example 4: Compare Portfolio Strategies
```python
from simulations import PortfolioSimulator

sim = PortfolioSimulator(data_dir='data')

# Energy sector
energy = ['ENB.TO', 'TRP.TO', 'CNQ.TO', 'SU.TO', 'CVE.TO']

# Different strategies
max_sharpe = sim.optimize_portfolio(energy, 'sharpe')
min_vol = sim.optimize_portfolio(energy, 'min_vol')
risk_parity = sim.risk_parity_portfolio(energy)

print("Strategy Comparison:")
print(f"\nMax Sharpe:  Return={max_sharpe['return']:.1f}%, Vol={max_sharpe['volatility']:.1f}%, Sharpe={max_sharpe['sharpe_ratio']:.3f}")
print(f"Min Vol:     Return={min_vol['return']:.1f}%, Vol={min_vol['volatility']:.1f}%, Sharpe={min_vol['sharpe_ratio']:.3f}")
print(f"Risk Parity: Return={risk_parity['return']:.1f}%, Vol={risk_parity['volatility']:.1f}%, Sharpe={risk_parity['sharpe_ratio']:.3f}")
```

### Example 5: Backtest a Portfolio
```python
from simulations import PortfolioSimulator

sim = PortfolioSimulator(data_dir='data')

# Your portfolio weights
portfolio = {
    'RY.TO': 0.30,   # 30% Royal Bank
    'ENB.TO': 0.25,  # 25% Enbridge
    'SHOP.TO': 0.20, # 20% Shopify
    'CNR.TO': 0.15,  # 15% Canadian National Railway
    'BCE.TO': 0.10   # 10% Bell Canada
}

# Backtest
results = sim.backtest_portfolio(portfolio)

print(f"Total Return: {results['total_return']:.2f}%")
print(f"Annual Return: {results['annual_return']:.2f}%")
print(f"Volatility: {results['annual_volatility']:.2f}%")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
```

### Example 6: Screen for Best Stocks
```python
import pandas as pd

# Load analysis results
df = pd.read_csv('tsx_analysis.csv')

# Find stocks with:
# - Sharpe ratio > 1.0
# - Annual return > 15%
# - Max drawdown > -30%
# - Beta < 1.2

screened = df[
    (df['sharpe_ratio'] > 1.0) &
    (df['annual_return'] > 15) &
    (df['max_drawdown'] > -30) &
    (df['beta'] < 1.2)
]

print("High-Quality Stocks:")
print(screened[['ticker', 'annual_return', 'sharpe_ratio', 'beta', 'max_drawdown']].to_string(index=False))
```

### Example 7: Sector Analysis
```python
from quant_analysis import QuantAnalyzer
import yfinance as yf

# Define sectors
sectors = {
    'Banks': ['RY.TO', 'TD.TO', 'BNS.TO', 'BMO.TO', 'CM.TO'],
    'Energy': ['ENB.TO', 'TRP.TO', 'CNQ.TO', 'SU.TO', 'CVE.TO'],
    'Telecom': ['BCE.TO', 'T.TO', 'RCI.B.TO'],
    'Railroads': ['CNR.TO', 'CP.TO']
}

analyzer = QuantAnalyzer(data_dir='data')
market = yf.Ticker('^GSPTSE').history(period='max')

for sector, tickers in sectors.items():
    returns = []
    sharpes = []
    betas = []

    for ticker in tickers:
        m = analyzer.analyze_stock(ticker, market)
        if m:
            returns.append(m['annual_return'])
            sharpes.append(m['sharpe_ratio'])
            betas.append(m['beta'])

    print(f"\n{sector}:")
    print(f"  Avg Return: {sum(returns)/len(returns):.2f}%")
    print(f"  Avg Sharpe: {sum(sharpes)/len(sharpes):.3f}")
    print(f"  Avg Beta: {sum(betas)/len(betas):.3f}")
```

## Output Files

| File | Description |
|------|-------------|
| `data/*.csv` | Historical price data for each stock |
| `data/*.json` | Company information |
| `tsx_analysis.csv` | Complete quantitative analysis for all stocks |
| `charts/*.png` | Visualization outputs |

## Metrics Explained

- **Annual Return**: Average yearly return (annualized)
- **Volatility**: Standard deviation of returns (risk measure)
- **Sharpe Ratio**: Risk-adjusted return (higher is better, >1 is good)
- **Sortino Ratio**: Like Sharpe but only considers downside risk
- **Beta**: Correlation with market (1 = moves with market)
- **Alpha**: Excess return vs. expected (positive = outperformance)
- **Max Drawdown**: Largest peak-to-trough decline
- **VaR (Value at Risk)**: Maximum expected loss at confidence level
- **CVaR**: Average loss beyond VaR threshold
- **Information Ratio**: Excess return per unit of tracking error
- **Calmar Ratio**: Return / Max Drawdown

## Tips

1. **Start Small**: Test with `quick_test.py` before downloading all data
2. **Data Quality**: Some TSX stocks may have limited historical data
3. **Diversification**: Use correlation matrix to build diversified portfolios
4. **Risk Management**: Always check max drawdown and VaR
5. **Rebalancing**: Re-run optimization monthly/quarterly
6. **Market Conditions**: Results based on historical data, past ≠ future

## Troubleshooting

**No data for ticker:**
- Verify ticker format (should end with .TO)
- Check if ticker exists on Yahoo Finance
- Try manually: `yfinance.Ticker('RY.TO').history(period='1d')`

**Analysis shows NaN/Inf:**
- Stock may have insufficient data
- Check for data quality issues
- Filter out in screening: `df[df['sharpe_ratio'].notna()]`

**Optimization fails:**
- Need at least 2 stocks
- Ensure all tickers have data
- Check for highly correlated stocks (use correlation matrix)

## Advanced Usage

### Custom Risk-Free Rate
```python
analyzer = QuantAnalyzer(data_dir='data')
analyzer.risk_free_rate = 0.05  # 5%
```

### Efficient Frontier
```python
from simulations import PortfolioSimulator

sim = PortfolioSimulator(data_dir='data')
tickers = ['RY.TO', 'ENB.TO', 'SHOP.TO', 'CNR.TO', 'BCE.TO']

# Generate efficient frontier
ef = sim.efficient_frontier(tickers, n_portfolios=5000)

# Visualize
from visualizations import Visualizer
viz = Visualizer(data_dir='data')
viz.plot_efficient_frontier(tickers)
```

## Next Steps

1. Run `python quick_test.py` to verify setup
2. Run `python main.py` for complete analysis
3. Explore individual modules
4. Customize for your needs
5. Build your own strategies!
