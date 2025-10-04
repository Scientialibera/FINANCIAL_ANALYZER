# TSX Stock Analysis System

A comprehensive quantitative analysis system for TSX (Toronto Stock Exchange) stocks using Yahoo Finance data.

## Features

### 1. **Data Download** (`tsx_data_downloader.py`)
- Downloads historical stock data for all TSX companies
- Scrapes company list from Wikipedia
- Fetches maximum available historical data from Yahoo Finance
- Saves data as CSV files and company info as JSON

### 2. **Quantitative Analysis** (`quant_analysis.py`)
Calculates comprehensive metrics for each stock:

**Return Metrics:**
- Total Return
- Annual Return
- Daily Return (mean/median)

**Volatility Metrics:**
- Daily & Annual Volatility
- Downside Deviation

**Risk-Adjusted Returns:**
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio

**Risk Metrics:**
- Beta (vs TSX Composite)
- Alpha
- Information Ratio
- Value at Risk (VaR) at 95% & 99%
- Conditional VaR (CVaR)
- Maximum Drawdown
- Ulcer Index

**Distribution:**
- Skewness
- Kurtosis

### 3. **Monte Carlo Simulations** (`simulations.py`)
- Price prediction using Geometric Brownian Motion
- Configurable forecast period and simulation count
- Probability distributions and confidence intervals

### 4. **Portfolio Optimization** (`simulations.py`)
- **Maximum Sharpe Ratio Portfolio**: Best risk-adjusted returns
- **Minimum Volatility Portfolio**: Lowest risk portfolio
- **Risk Parity Portfolio**: Equal risk contribution from each asset
- Efficient Frontier generation
- Portfolio backtesting

### 5. **Visualizations** (`visualizations.py`)
- Monte Carlo simulation plots
- Efficient Frontier charts
- Portfolio strategy comparisons
- Correlation matrices
- Top performers analysis

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Option 1: Run Complete Analysis (Interactive)
```bash
python main.py
```

### Option 2: Run Individual Modules

#### Download TSX Data
```bash
python tsx_data_downloader.py
```

#### Run Quantitative Analysis
```bash
python quant_analysis.py
```

#### Run Monte Carlo Simulation
```python
from simulations import PortfolioSimulator

sim = PortfolioSimulator(data_dir='data')
results = sim.monte_carlo_simulation('RY.TO', days=252, simulations=10000)
print(f"Mean predicted price: ${results['mean_final_price']:.2f}")
print(f"Probability of profit: {results['probability_profit']:.1f}%")
```

#### Optimize Portfolio
```python
from simulations import PortfolioSimulator

sim = PortfolioSimulator(data_dir='data')
tickers = ['RY.TO', 'TD.TO', 'BNS.TO', 'BMO.TO', 'CM.TO']

# Maximum Sharpe Ratio
max_sharpe = sim.optimize_portfolio(tickers, objective='sharpe')
print(f"Expected Return: {max_sharpe['return']:.2f}%")
print(f"Sharpe Ratio: {max_sharpe['sharpe_ratio']:.3f}")

# Risk Parity
risk_parity = sim.risk_parity_portfolio(tickers)
```

#### Generate Visualizations
```bash
python visualizations.py
```

## Output Files

- `data/`: Historical stock data (CSV) and company info (JSON)
- `tsx_analysis.csv`: Comprehensive quantitative metrics for all stocks
- `charts/`: Visualization outputs (PNG)

## Yahoo Finance Ticker Format

TSX stocks use `.TO` suffix (e.g., `RY.TO` for Royal Bank of Canada)

## Examples

### Example 1: Analyze Top Banks
```python
from quant_analysis import QuantAnalyzer

analyzer = QuantAnalyzer(data_dir='data')
banks = ['RY.TO', 'TD.TO', 'BNS.TO', 'BMO.TO', 'CM.TO']

for bank in banks:
    metrics = analyzer.analyze_stock(bank)
    print(f"{bank}: Sharpe={metrics['sharpe_ratio']:.3f}, Return={metrics['annual_return']:.2f}%")
```

### Example 2: Monte Carlo Forecast
```python
from simulations import PortfolioSimulator

sim = PortfolioSimulator(data_dir='data')
mc = sim.monte_carlo_simulation('SHOP.TO', days=252, simulations=10000)

print(f"Current: ${mc['current_price']:.2f}")
print(f"1-Year Forecast (median): ${mc['median_final_price']:.2f}")
print(f"95% Confidence: ${mc['percentile_5']:.2f} - ${mc['percentile_95']:.2f}")
```

### Example 3: Build Optimal Portfolio
```python
from simulations import PortfolioSimulator

sim = PortfolioSimulator(data_dir='data')

# Energy sector stocks
energy = ['ENB.TO', 'TRP.TO', 'CNQ.TO', 'SU.TO', 'CVE.TO']

# Find maximum Sharpe ratio portfolio
optimal = sim.optimize_portfolio(energy, objective='sharpe')

print("Optimal Weights:")
for ticker, weight in optimal['weights'].items():
    print(f"  {ticker}: {weight*100:.1f}%")
```

## Modules

| Module | Description |
|--------|-------------|
| `tsx_data_downloader.py` | Download TSX stock data from Yahoo Finance |
| `quant_analysis.py` | Quantitative analysis and metrics calculation |
| `simulations.py` | Monte Carlo simulations and portfolio optimization |
| `visualizations.py` | Create charts and visualizations |
| `main.py` | Interactive main script (runs complete pipeline) |
| `test_tsx_quotes.py` | Simple test script to verify API access |

## Technical Details

### Calculations

**Sharpe Ratio:**
```
Sharpe = (Return - Risk_Free_Rate) / Volatility
```

**Beta:**
```
Beta = Covariance(Stock, Market) / Variance(Market)
```

**Alpha:**
```
Alpha = Stock_Return - [Risk_Free_Rate + Beta × (Market_Return - Risk_Free_Rate)]
```

**Monte Carlo (Geometric Brownian Motion):**
```
S(t+1) = S(t) × exp[(μ - σ²/2)Δt + σ√Δt × Z]
where Z ~ N(0,1)
```

### Assumptions
- Risk-free rate: 4.5% (approximate current Canadian rate)
- Trading days per year: 252
- Benchmark index: ^GSPTSE (S&P/TSX Composite Index)

## License

This project is for educational and research purposes.

## Disclaimer

This software is for informational purposes only. It is not financial advice. Always do your own research and consult with a qualified financial advisor before making investment decisions.
