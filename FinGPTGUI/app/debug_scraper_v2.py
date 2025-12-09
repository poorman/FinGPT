import pandas as pd
import requests

print("Testing S&P 500 Scraper (v2)...")
try:
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    print(f"Reading {url} with User-Agent...")
    response = requests.get(url, headers=headers)
    
    tables = pd.read_html(response.text)
    print(f"Found {len(tables)} tables.")
    df = tables[0]
    tickers = df['Symbol'].tolist()
    print(f"Successfully scraped {len(tickers)} tickers. First 5: {tickers[:5]}")
except Exception as e:
    print(f"CRASH: {e}")
