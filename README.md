# TSX Financial Analyzer

**Enterprise-Grade Quantitative Analysis System for TSX Stocks**

---

## Overview

Professional quantitative analysis system for Toronto Stock Exchange (TSX) stocks featuring:

- **Automated Data Collection**: Programmatic ticker scraping from Wikipedia + Yahoo Finance API
- **Advanced Analytics**: 20+ quantitative metrics (Sharpe, Sortino, Alpha, Beta, VaR, CVaR, etc.)
- **Monte Carlo Simulations**: 5 methods (GBM, Jump Diffusion, Bootstrap, Regime Switching, GARCH)
- **Portfolio Optimization**: Modern Portfolio Theory with multiple strategies
- **Enterprise Architecture**: Proper logging, timestamped outputs, caching, error handling

---

## Project Structure

```
FINANCIAL_ANALYZER/
+-- analyze.py              # MAIN EXECUTABLE - Run this!
+-- requirements.txt        # Python dependencies
+-- README.md              # This file
|
+-- config/                # Configuration
|   +-- __init__.py
|   +-- settings.py        # All parameters
|
+-- src/                   # Source modules
|   +-- ticker_scraper.py
|   +-- data_downloader.py
|   +-- analyzer.py
|   +-- portfolio_optimizer.py
|   +-- monte_carlo.py
|
+-- output/                # ALL outputs
|   +-- data/             # Stock data
|   +-- reports/          # Analysis CSVs (timestamped)
|   +-- portfolios/       # Portfolio recommendations
|
+-- cache/                 # Cached ticker lists
+-- logs/                  # Execution logs
```

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```bash
# FULL ANALYSIS - All TSX stocks (recommended!)
python analyze.py --full

# Quick test with 20 stocks
python analyze.py --quick

# Quick test with 50 stocks
python analyze.py --quick --stocks 50

# Portfolio optimization only
python analyze.py --portfolio

# Monte Carlo simulation
python analyze.py --simulate RY.TO
python analyze.py --simulate SHOP.TO --days 365 --sims 20000
```

---

## Command Reference

### Main Commands

| Command | Description |
|---------|-------------|
| `--full` | Complete pipeline: scrape -> download -> analyze -> optimize |
| `--quick` | Quick test with limited stocks (default 20) |
| `--portfolio` | Portfolio optimization only (uses cached analysis) |
| `--simulate TICKER` | Monte Carlo simulation for specific ticker |

### Options

| Option | Description | Example |
|--------|-------------|---------|
| `--stocks N` | Limit number of stocks to download | `--stocks 100` |
| `--num-stocks N` | Number of stocks in portfolio | `--num-stocks 15` |
| `--min-sharpe X` | Minimum Sharpe ratio threshold | `--min-sharpe 0.5` |
| `--days N` | Simulation days ahead | `--days 365` |
| `--sims N` | Number of simulations | `--sims 20000` |

### Examples

```bash
# Analyze top 100 stocks, create 15-stock portfolio
python analyze.py --full --stocks 100 --num-stocks 15

# Portfolio with high Sharpe threshold
python analyze.py --portfolio --num-stocks 10 --min-sharpe 0.8

# 2-year price forecast
python analyze.py --simulate SHOP.TO --days 504 --sims 50000

# Quick test
python analyze.py --quick --stocks 10
```

---

## Features

### 1. Automated Data Collection

- Ticker Scraping: Automatically scrapes ALL TSX tickers from Wikipedia
- Yahoo Finance Download: Complete historical data for each stock
- Smart Formatting: Handles .TO suffix, dual-class shares (.A, .B), REITs (.UN)
- Validation: Tests each ticker with Yahoo Finance API
- Caching: Saves ticker lists for reuse

### 2. Quantitative Analysis (20+ Metrics)

| Category | Metrics |
|----------|---------|
| **Returns** | Total, Annual, Daily Mean/Median |
| **Volatility** | Annual/Daily Volatility, Downside Deviation |
| **Risk-Adjusted** | Sharpe, Sortino, Calmar, Information Ratio |
| **Market** | Beta, Alpha (vs ^GSPTSE) |
| **Risk** | Max Drawdown, VaR (95%/99%), CVaR, Ulcer Index |
| **Distribution** | Skewness, Kurtosis |

### 3. Monte Carlo Simulations (5 Methods)

1. **Geometric Brownian Motion (GBM)** - Classic Black-Scholes model
2. **GBM with Jump Diffusion** - Merton model, captures crashes/spikes
3. **Historical Bootstrap** - Samples actual returns, non-parametric
4. **Regime Switching** - Bull/bear market transitions
5. **GARCH Volatility** - Time-varying volatility clustering

### 4. Portfolio Optimization (4 Strategies)

- **Maximum Sharpe Ratio** - Best risk-adjusted returns
- **Minimum Volatility** - Lowest risk
- **Risk Parity** - Equal risk contribution
- **Equal Weight** - Simple baseline

---

## Output Files

All outputs are **automatically timestamped** and organized:

### Reports (output/reports/)

```
tsx_analysis_20250104_143022.csv          # Complete analysis
optimal_portfolio_20250104_143525.csv     # Portfolio recommendations
analysis_report_20250104_143525.txt       # Text summary
```

### Data (output/data/)

```
RY_TO_history.csv     # Historical OHLCV data
RY_TO_info.json       # Company fundamentals
```

### Cache (cache/)

```
tsx_tickers_20250104_142800.csv    # Scraped ticker list
```

### Logs (logs/)

```
analysis_20250104.log               # Daily log (detailed debugging)
download_log_20250104_143022.csv    # Download success/failure tracking
```

---

## Configuration

Edit `config/settings.py` to customize:

```python
# Analysis parameters
ANALYSIS = {
    'risk_free_rate': 0.045,           # 4.5% Canadian T-bills
    'benchmark_ticker': '^GSPTSE',     # TSX Composite Index
    'min_sharpe_threshold': 0.3,       # Portfolio inclusion
}

# Portfolio constraints
PORTFOLIO = {
    'default_stocks': 10,
    'max_weight': 0.40,    # Max 40% in single position
    'min_weight': 0.01,    # Min 1% allocation
}

# Download settings
DOWNLOAD = {
    'yahoo_delay': 0.5,        # Seconds between API calls
    'pause_every': 50,         # Pause after N downloads
}
```

---

## Important Notes

### Disclaimers

1. **Not Financial Advice** - Educational/research purposes only
2. **Past != Future** - Historical data doesn't guarantee future results
3. **Risk** - All investments carry risk of loss
4. **Professional Advice** - Consult qualified financial advisor
5. **Data Quality** - Yahoo Finance may have errors/gaps

### Technical Notes

- **Risk-Free Rate**: 4.5% (Canadian T-bills)
- **Trading Days**: 252 per year
- **Benchmark**: S&P/TSX Composite (^GSPTSE)
- **No Costs**: Analysis excludes fees/commissions
- **No Taxes**: Consider tax implications separately

---

## System Requirements

- **Python**: 3.10+
- **RAM**: 4GB+ recommended
- **Disk**: 500MB for data storage
- **Internet**: Required for data download

### Dependencies

```
yfinance          # Yahoo Finance API
pandas            # Data manipulation
numpy             # Numerical computing
scipy             # Optimization & statistics
beautifulsoup4    # Web scraping
matplotlib        # Visualization
requests          # HTTP requests
```

Install all:
```bash
pip install -r requirements.txt
```

---

## Methodology

### Portfolio Optimization

Uses **Modern Portfolio Theory** (Markowitz):
- Objective: Maximize Sharpe ratio
- Constraints: No shorting, weights sum to 1
- Method: Scipy SLSQP optimizer

### Monte Carlo

**Geometric Brownian Motion**:
```
S(t+1) = S(t) x exp[(mu - sigma^2/2)dt + sigma*sqrt(dt)*Z]
where Z ~ N(0,1), mu = drift, sigma = volatility
```

**Regime Switching**:
- Bull regime: Low volatility, positive drift
- Bear regime: High volatility, negative drift
- Transition probabilities: 5% bull->bear, 15% bear->bull

---

## Support

For issues:
1. Check `logs/analysis_YYYYMMDD.log`
2. Verify ticker format (needs .TO suffix)
3. Confirm Yahoo Finance has data
4. Review `config/settings.py`

---

## License

MIT License - For educational purposes only

---

## Credits

- Data: Yahoo Finance API
- Tickers: Wikipedia
- Built with: Python + NumPy + SciPy + Pandas

---

## Version

**v2.0.0** - Enterprise Edition

---

**Quick Start**: `python analyze.py --full`

**Built with Python | Powered by Modern Portfolio Theory | For TSX Analysis**
