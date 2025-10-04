"""
Comprehensive TSX Ticker Scraper
Downloads ALL TSX companies from multiple sources and validates them
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import yfinance as yf
import time
import re
from datetime import datetime

class TSXTickerScraper:
    def __init__(self):
        self.tickers = set()
        self.ticker_details = []

    def scrape_wikipedia_tsx_composite(self):
        """Scrape S&P/TSX Composite Index from Wikipedia"""
        print("\n[1] Scraping S&P/TSX Composite Index from Wikipedia...")

        url = "https://en.wikipedia.org/wiki/S%26P/TSX_Composite_Index"
        headers = {'User-Agent': 'Mozilla/5.0'}

        try:
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find all tables
            tables = soup.find_all('table', {'class': 'wikitable'})

            count = 0
            for table in tables:
                rows = table.find_all('tr')[1:]  # Skip header
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 2:
                        # Try to extract ticker symbol (usually in first few columns)
                        for i in range(min(3, len(cols))):
                            text = cols[i].text.strip()
                            # Look for ticker pattern (2-5 letters, possibly with periods)
                            if re.match(r'^[A-Z]{1,5}(\.[A-Z])?$', text):
                                ticker = text
                                # Get company name (usually in adjacent column)
                                company = cols[1].text.strip() if i == 0 else cols[0].text.strip()

                                self.tickers.add(ticker)
                                self.ticker_details.append({
                                    'ticker': ticker,
                                    'company': company,
                                    'source': 'Wikipedia_TSX_Composite'
                                })
                                count += 1
                                break

            print(f"   Found {count} companies from TSX Composite")

        except Exception as e:
            print(f"   Error: {e}")

    def scrape_wikipedia_tsx60(self):
        """Scrape S&P/TSX 60 from Wikipedia"""
        print("\n[2] Scraping S&P/TSX 60 from Wikipedia...")

        url = "https://en.wikipedia.org/wiki/S%26P/TSX_60"
        headers = {'User-Agent': 'Mozilla/5.0'}

        try:
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.content, 'html.parser')

            tables = soup.find_all('table', {'class': 'wikitable'})

            count = 0
            for table in tables:
                rows = table.find_all('tr')[1:]
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 2:
                        for i in range(min(3, len(cols))):
                            text = cols[i].text.strip()
                            if re.match(r'^[A-Z]{1,5}(\.[A-Z])?$', text):
                                ticker = text
                                company = cols[1].text.strip() if i == 0 else cols[0].text.strip()

                                if ticker not in self.tickers:
                                    self.tickers.add(ticker)
                                    self.ticker_details.append({
                                        'ticker': ticker,
                                        'company': company,
                                        'source': 'Wikipedia_TSX60'
                                    })
                                    count += 1
                                break

            print(f"   Found {count} new companies from TSX 60")

        except Exception as e:
            print(f"   Error: {e}")

    def scrape_tmx_listings(self):
        """
        Scrape TMX (Toronto Stock Exchange) official listings
        Note: This uses a fallback approach as TMX may require API access
        """
        print("\n[3] Attempting to get TMX official listings...")

        # TMX Money has a public endpoint (may change)
        try:
            url = "https://www.tmxmoney.com/en/index.html"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)

            # This would need to be adapted based on TMX's actual structure
            print("   TMX direct scraping not implemented (requires API or more complex scraping)")
            print("   Using Wikipedia sources instead")

        except Exception as e:
            print(f"   TMX scraping skipped: {e}")

    def add_known_major_companies(self):
        """Add known major TSX companies that might be missed"""
        print("\n[4] Adding known major TSX companies...")

        major_companies = {
            # Big Banks
            'RY': 'Royal Bank of Canada',
            'TD': 'Toronto-Dominion Bank',
            'BNS': 'Bank of Nova Scotia',
            'BMO': 'Bank of Montreal',
            'CM': 'CIBC',
            'NA': 'National Bank of Canada',

            # Energy
            'ENB': 'Enbridge',
            'TRP': 'TC Energy',
            'CNQ': 'Canadian Natural Resources',
            'SU': 'Suncor Energy',
            'CVE': 'Cenovus Energy',
            'IMO': 'Imperial Oil',

            # Telecom
            'T': 'Telus',
            'BCE': 'Bell Canada',
            'RCI.B': 'Rogers Communications',

            # Retail/Consumer
            'L': 'Loblaw',
            'ATD': 'Alimentation Couche-Tard',
            'DOL': 'Dollarama',
            'QSR': 'Restaurant Brands International',

            # Tech
            'SHOP': 'Shopify',
            'BB': 'BlackBerry',

            # Industrials
            'CNR': 'Canadian National Railway',
            'CP': 'Canadian Pacific Railway',

            # Mining/Materials
            'ABX': 'Barrick Gold',
            'NTR': 'Nutrien',
            'FNV': 'Franco-Nevada',
            'WPM': 'Wheaton Precious Metals',

            # Utilities
            'FTS': 'Fortis',
            'EMA': 'Emera',
            'H': 'Hydro One',

            # Insurance
            'MFC': 'Manulife',
            'SLF': 'Sun Life Financial',
            'IFC': 'Intact Financial',

            # Real Estate
            'BPY.UN': 'Brookfield Property Partners',
        }

        count = 0
        for ticker, company in major_companies.items():
            if ticker not in self.tickers:
                self.tickers.add(ticker)
                self.ticker_details.append({
                    'ticker': ticker,
                    'company': company,
                    'source': 'Major_Known_Companies'
                })
                count += 1

        print(f"   Added {count} major companies")

    def format_ticker_for_yahoo(self, ticker):
        """
        Format ticker for Yahoo Finance API
        TSX stocks need .TO suffix
        """
        ticker = ticker.strip().upper()

        # If already has .TO, return as is
        if ticker.endswith('.TO'):
            return ticker

        # Special cases for dual-class shares
        if ticker.endswith('.A') or ticker.endswith('.B'):
            # Remove the class suffix and add .TO
            base = ticker[:-2]
            class_suffix = ticker[-2:]
            return f"{base}{class_suffix}.TO"

        # For REITs with .UN
        if ticker.endswith('.UN'):
            return ticker + '.TO'

        # Standard case: just add .TO
        return ticker + '.TO'

    def validate_tickers(self, sample_size=None):
        """
        Validate tickers by attempting to fetch data from Yahoo Finance
        """
        print("\n[5] Validating tickers with Yahoo Finance...")

        tickers_to_validate = list(self.tickers)
        if sample_size:
            import random
            tickers_to_validate = random.sample(tickers_to_validate, min(sample_size, len(tickers_to_validate)))
            print(f"   Validating sample of {len(tickers_to_validate)} tickers...")
        else:
            print(f"   Validating all {len(tickers_to_validate)} tickers...")

        valid_tickers = []
        invalid_tickers = []

        for i, ticker in enumerate(tickers_to_validate, 1):
            if i % 10 == 0:
                print(f"   Progress: {i}/{len(tickers_to_validate)}", end='\r')

            yahoo_ticker = self.format_ticker_for_yahoo(ticker)

            try:
                stock = yf.Ticker(yahoo_ticker)
                # Try to get basic info
                info = stock.info
                hist = stock.history(period='5d')

                if not hist.empty and len(hist) > 0:
                    valid_tickers.append({
                        'tsx_ticker': ticker,
                        'yahoo_ticker': yahoo_ticker,
                        'name': info.get('longName', info.get('shortName', 'Unknown')),
                        'sector': info.get('sector', 'Unknown'),
                        'industry': info.get('industry', 'Unknown'),
                        'market_cap': info.get('marketCap', 0)
                    })
                else:
                    invalid_tickers.append(ticker)

            except Exception as e:
                invalid_tickers.append(ticker)

            time.sleep(0.2)  # Be nice to Yahoo Finance API

        print(f"\n   Valid: {len(valid_tickers)}, Invalid: {len(invalid_tickers)}")

        return valid_tickers, invalid_tickers

    def get_all_tsx_tickers(self, validate=True, validation_sample=50):
        """
        Main method to get all TSX tickers
        """
        print("=" * 80)
        print("TSX TICKER SCRAPER - Getting ALL TSX Companies")
        print("=" * 80)

        # Scrape from multiple sources
        self.scrape_wikipedia_tsx_composite()
        self.scrape_wikipedia_tsx60()
        self.add_known_major_companies()

        print(f"\n{'='*80}")
        print(f"Total unique tickers found: {len(self.tickers)}")
        print("=" * 80)

        # Format all tickers for Yahoo Finance
        formatted_tickers = []
        for ticker in self.tickers:
            yahoo_ticker = self.format_ticker_for_yahoo(ticker)
            formatted_tickers.append({
                'tsx_ticker': ticker,
                'yahoo_ticker': yahoo_ticker
            })

        # Validate if requested
        if validate:
            valid_tickers, invalid_tickers = self.validate_tickers(sample_size=validation_sample)

            # Save results
            if valid_tickers:
                df_valid = pd.DataFrame(valid_tickers)
                df_valid.to_csv('tsx_all_tickers_validated.csv', index=False)
                print(f"\nSaved validated tickers to: tsx_all_tickers_validated.csv")

        # Save all tickers (including unvalidated)
        df_all = pd.DataFrame(formatted_tickers)
        df_all.to_csv('tsx_all_tickers.csv', index=False)
        print(f"Saved all tickers to: tsx_all_tickers.csv")

        return formatted_tickers

def main():
    scraper = TSXTickerScraper()

    # Get all tickers and validate a sample
    tickers = scraper.get_all_tsx_tickers(validate=True, validation_sample=50)

    print("\n" + "=" * 80)
    print("SAMPLE OF TICKERS:")
    print("=" * 80)

    # Display sample
    df = pd.DataFrame(tickers)
    print(df.head(20).to_string(index=False))

    print(f"\nTotal tickers with Yahoo Finance format: {len(tickers)}")
    print("\nNext steps:")
    print("1. Review tsx_all_tickers.csv")
    print("2. Review tsx_all_tickers_validated.csv (sample validation)")
    print("3. Use these tickers for full download and analysis")

if __name__ == "__main__":
    main()
