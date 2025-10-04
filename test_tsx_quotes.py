import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup

def get_tsx_companies():
    """
    Get list of TSX companies from Wikipedia
    """
    # Try multiple sources
    url1 = "https://en.wikipedia.org/wiki/S%26P/TSX_Composite_Index"
    url2 = "https://en.wikipedia.org/wiki/S%26P/TSX_60"

    companies = set()

    try:
        # Try main TSX Composite page
        response = requests.get(url1, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all tables
        tables = soup.find_all('table', {'class': 'wikitable'})

        for table in tables:
            rows = table.find_all('tr')[1:]  # Skip header
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 1:
                    # Try each column to find ticker
                    for col in cols[:3]:  # Check first 3 columns
                        text = col.text.strip()
                        # Look for ticker patterns (all caps, short length)
                        if text and len(text) <= 6 and text.replace('.', '').replace('-', '').isalnum():
                            # Skip if it's a number or date
                            if not text.replace(',', '').replace('.', '').isdigit():
                                companies.add(text + '.TO')
    except Exception as e:
        print(f"Error fetching from Wikipedia: {e}")

    # If we didn't get many companies, use fallback list of major TSX companies
    if len(companies) < 10:
        print("Using fallback list of major TSX companies")
        companies = {
            'RY.TO', 'TD.TO', 'BNS.TO', 'BMO.TO', 'CM.TO',  # Banks
            'ENB.TO', 'TRP.TO', 'CNQ.TO', 'SU.TO', 'CVE.TO',  # Energy
            'SHOP.TO', 'BCE.TO', 'T.TO',  # Tech & Telecom
            'CNR.TO', 'CP.TO',  # Railroads
            'ABX.TO', 'GOLD.TO',  # Mining
            'MFC.TO', 'SLF.TO',  # Insurance
            'WN.TO', 'L.TO'  # Retail
        }

    return sorted(list(companies))

def test_yahoo_finance(ticker):
    """
    Test fetching a quote from Yahoo Finance
    """
    print(f"\nTesting Yahoo Finance API with ticker: {ticker}")
    print("=" * 60)

    try:
        stock = yf.Ticker(ticker)

        # Get basic info
        info = stock.info
        print(f"\nCompany: {info.get('longName', 'N/A')}")
        print(f"Ticker: {ticker}")
        print(f"Current Price: ${info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))}")
        print(f"Market Cap: ${info.get('marketCap', 'N/A'):,}" if info.get('marketCap') else "Market Cap: N/A")

        # Get historical data
        hist = stock.history(period="1mo")
        print(f"\nHistorical Data (last 5 days):")
        print(hist.tail())

        return True
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return False

if __name__ == "__main__":
    print("Fetching TSX company listings...")
    companies = get_tsx_companies()
    print(f"Found {len(companies)} TSX companies")
    print(f"Sample tickers: {companies[:10]}")

    # Test with first available ticker
    if companies:
        test_yahoo_finance(companies[0])
