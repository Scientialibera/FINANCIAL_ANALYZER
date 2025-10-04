"""
Configuration settings for TSX Financial Analyzer
"""
import os
from pathlib import Path
from datetime import datetime

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Directory structure
DIRS = {
    'output': BASE_DIR / 'output',
    'logs': BASE_DIR / 'logs',
    'cache': BASE_DIR / 'cache',
    'data': BASE_DIR / 'output' / 'data',
    'reports': BASE_DIR / 'output' / 'reports',
    'portfolios': BASE_DIR / 'output' / 'portfolios',
}

# Create directories
for dir_path in DIRS.values():
    dir_path.mkdir(parents=True, exist_ok=True)

# Analysis settings
ANALYSIS = {
    'risk_free_rate': 0.045,  # 4.5% Canadian T-bills
    'benchmark_ticker': '^GSPTSE',  # TSX Composite Index
    'trading_days_per_year': 252,
    'min_data_points': 100,  # Minimum days of data required
    'min_sharpe_threshold': 0.3,  # Minimum Sharpe for portfolio inclusion
}

# Data download settings
DOWNLOAD = {
    'yahoo_delay': 0.5,  # Seconds between API calls
    'pause_every': 50,  # Pause after N downloads
    'pause_duration': 2,  # Pause duration in seconds
    'max_retries': 3,
    'timeout': 30,
}

# Portfolio optimization settings
PORTFOLIO = {
    'default_stocks': 10,
    'max_weight': 0.40,  # Maximum 40% in single stock
    'min_weight': 0.01,  # Minimum 1% allocation
}

# Simulation settings
SIMULATION = {
    'default_simulations': 10000,
    'default_days': 252,  # 1 year
    'jump_probability': 0.02,  # 2% chance of jump per day
}

# Output file naming
def get_output_filename(prefix, extension='csv'):
    """Generate timestamped output filename"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return DIRS['reports'] / f"{prefix}_{timestamp}.{extension}"

def get_data_cache_path(ticker):
    """Get cache path for ticker data"""
    clean_ticker = ticker.replace('.', '_')
    return DIRS['cache'] / f"{clean_ticker}_data.csv"

def get_log_file():
    """Get current log file path"""
    date_str = datetime.now().strftime('%Y%m%d')
    return DIRS['logs'] / f"analysis_{date_str}.log"

# Logging settings
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'simple': {
            'format': '%(levelname)s - %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'simple',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.FileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': str(get_log_file()),
            'mode': 'a'
        }
    },
    'root': {
        'level': 'DEBUG',
        'handlers': ['console', 'file']
    }
}
