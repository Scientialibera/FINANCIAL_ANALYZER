"""
TSX Stock Data Downloader
Downloads all available historical data for TSX companies from Yahoo Finance
"""

import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime
import os
import json

class TSXDataDownloader:
    def __init__(self, output_dir='data'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def get_tsx_companies(self):
        """
        Get list of TSX companies from Wikipedia
        Returns list of ticker symbols with .TO suffix
        """
        url = "https://en.wikipedia.org/wiki/S%26P/TSX_Composite_Index"
        companies = set()

        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(response.content, 'html.parser')
            tables = soup.find_all('table', {'class': 'wikitable'})

            for table in tables:
                rows = table.find_all('tr')[1:]
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 1:
                        for col in cols[:3]:
                            text = col.text.strip()
                            if text and len(text) <= 6 and text.replace('.', '').replace('-', '').isalnum():
                                if not text.replace(',', '').replace('.', '').isdigit():
                                    companies.add(text + '.TO')
        except Exception as e:
            print(f"Error fetching from Wikipedia: {e}")

        # Fallback list if scraping fails
        if len(companies) < 10:
            print("Using fallback list of major TSX companies")
            companies = {
                'RY.TO', 'TD.TO', 'BNS.TO', 'BMO.TO', 'CM.TO',
                'ENB.TO', 'TRP.TO', 'CNQ.TO', 'SU.TO', 'CVE.TO',
                'SHOP.TO', 'BCE.TO', 'T.TO', 'CNR.TO', 'CP.TO',
                'ABX.TO', 'GOLD.TO', 'MFC.TO', 'SLF.TO', 'WN.TO', 'L.TO'
            }

        return sorted(list(companies))

    def download_stock_data(self, ticker, period='max'):
        """
        Download historical data for a single stock
        """
        try:
            print(f"Downloading {ticker}...", end=' ')
            stock = yf.Ticker(ticker)

            # Get historical data for maximum available period
            hist = stock.history(period=period)

            if hist.empty:
                print(f"No data available")
                return None

            # Get company info
            info = stock.info

            # Save historical data
            csv_path = os.path.join(self.output_dir, f"{ticker.replace('.', '_')}_history.csv")
            hist.to_csv(csv_path)

            # Save company info
            info_path = os.path.join(self.output_dir, f"{ticker.replace('.', '_')}_info.json")
            with open(info_path, 'w') as f:
                json.dump(info, f, indent=2, default=str)

            print(f"OK ({len(hist)} days, from {hist.index[0].date()} to {hist.index[-1].date()})")
            return hist

        except Exception as e:
            print(f"Error: {str(e)}")
            return None

    def download_all_tsx_stocks(self):
        """
        Download data for all TSX companies
        """
        companies = self.get_tsx_companies()
        print(f"Found {len(companies)} TSX companies")
        print("=" * 80)

        successful = 0
        failed = 0

        for i, ticker in enumerate(companies, 1):
            print(f"[{i}/{len(companies)}] ", end='')
            result = self.download_stock_data(ticker)

            if result is not None:
                successful += 1
            else:
                failed += 1

            # Be nice to Yahoo Finance API
            time.sleep(0.5)

        print("=" * 80)
        print(f"Download complete!")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Data saved to: {os.path.abspath(self.output_dir)}")

if __name__ == "__main__":
    downloader = TSXDataDownloader(output_dir='data')
    downloader.download_all_tsx_stocks()
